import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, cast, TypeVar, Hashable
import matplotlib.pyplot as plt
from scipy import signal as sig_proc

# Importaciones locales (asumidas)
from core.quantum.circuit import ResilientQuantumCircuit
from core.analysis.bayes_logic import BayesLogic
from core.analysis.statistical_analysis import StatisticalAnalysis

# Configuración del logger
logger = logging.getLogger(__name__)

# Definición de tipos para anotaciones
QuantumState = List[complex]
ProcessedResult = Dict[str, Union[np.ndarray, float]]
T = TypeVar('T')  # Tipo genérico para funciones cache

class FFTBayesIntegrator:
    """
    Clase que integra la Transformada Rápida de Fourier (FFT) con el análisis bayesiano
    para procesar señales cuánticas y generar representaciones para la inicialización informada
    de modelos o como features para redes neuronales.
    
    Esta clase combina técnicas de análisis de señales (FFT) con inferencia bayesiana para 
    extraer y procesar información relevante de estados cuánticos. Permite realizar análisis 
    de características frecuenciales, calcular medidas de entropía y coherencia, y generar
    inicializadores para redes neuronales basados en propiedades cuánticas.
    
    Atributos:
        bayes_logic: Instancia de BayesLogic para cálculos probabilísticos.
        stat_analysis: Instancia de StatisticalAnalysis para análisis estadísticos.
        cache: Diccionario que almacena resultados previos para evitar recálculos.
        _fft_cache_hits: Contador de aciertos de caché (para diagnóstico).
        _fft_cache_misses: Contador de fallos de caché (para diagnóstico).
        
    Ejemplo de uso:
        >>> circuit = ResilientQuantumCircuit(...)  # Crear un circuito cuántico
        >>> integrator = FFTBayesIntegrator()
        >>> result = integrator.process_quantum_circuit(circuit)
        >>> print(f"Entropía del estado: {result['entropy']}")
        >>> print(f"Coherencia del estado: {result['coherence']}")
    """
    
    def __init__(self, cache_size: int = 1000) -> None:
        """
        Inicializa el integrador FFT-Bayes con componentes para análisis estadístico y bayesiano.
        
        Args:
            cache_size: Tamaño máximo de la caché para almacenar resultados (0 para desactivar).
        """
        # Inicializa instancias de lógica bayesiana y análisis estadístico
        self.bayes_logic = BayesLogic()
        self.stat_analysis = StatisticalAnalysis()
        
        # Inicializa la caché y contadores
        self.cache: Dict[int, ProcessedResult] = {}
        self._cache_size = cache_size
        self._fft_cache_hits = 0
        self._fft_cache_misses = 0
        
        logger.info(f"FFTBayesIntegrator inicializado con caché de tamaño {cache_size}")

    def _cached_compute(self, key: Hashable, compute_func: callable, *args, **kwargs) -> Any:
        """
        Mecanismo genérico de caché para evitar recálculos.
        
        Args:
            key: Clave única para identificar el cálculo.
            compute_func: Función que realiza el cálculo real.
            *args, **kwargs: Argumentos para la función de cálculo.
            
        Returns:
            El resultado del cálculo, ya sea de la caché o recién calculado.
        """
        # Si la caché está desactivada, simplemente ejecuta la función
        if self._cache_size == 0:
            return compute_func(*args, **kwargs)
        
        # Intenta obtener el resultado de la caché
        key_hash = hash(key)
        if key_hash in self.cache:
            self._fft_cache_hits += 1
            logger.debug(f"Acierto de caché para clave {key_hash} (hits: {self._fft_cache_hits})")
            return self.cache[key_hash]
        
        # Calcular el resultado y almacenarlo en caché
        self._fft_cache_misses += 1
        result = compute_func(*args, **kwargs)
        
        # Gestión de tamaño de caché (FIFO simple)
        if len(self.cache) >= self._cache_size:
            # Eliminar la entrada más antigua
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Caché llena, eliminando clave {oldest_key}")
        
        self.cache[key_hash] = result
        logger.debug(f"Fallo de caché para clave {key_hash} (misses: {self._fft_cache_misses})")
        return result

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Obtiene estadísticas sobre el rendimiento de la caché.
        
        Returns:
            Diccionario con estadísticas (hits, misses, tamaño actual, tamaño máximo).
        """
        return {
            'hits': self._fft_cache_hits,
            'misses': self._fft_cache_misses,
            'current_size': len(self.cache),
            'max_size': self._cache_size,
            'hit_ratio': self._fft_cache_hits / max(1, (self._fft_cache_hits + self._fft_cache_misses))
        }

    def clear_cache(self) -> None:
        """Limpia la caché y reinicia contadores."""
        self.cache.clear()
        self._fft_cache_hits = 0
        self._fft_cache_misses = 0
        logger.info("Caché y contadores reiniciados")

    def process_quantum_circuit(self, quantum_circuit: ResilientQuantumCircuit) -> ProcessedResult:
        """
        Procesa un circuito cuántico resistente aplicando la FFT a su estado.
        
        Este método extrae las amplitudes complejas del circuito cuántico y las
        procesa mediante análisis FFT para obtener características frecuenciales.
        
        Args:
            quantum_circuit: Instancia de un circuito cuántico del que extraer amplitudes.
            
        Returns:
            Diccionario con los resultados del análisis (magnitudes, fases, entropía, coherencia).
            
        Raises:
            ValueError: Si el circuito no tiene un estado válido.
        """
        if quantum_circuit is None:
            raise ValueError("El circuito cuántico no puede ser None")
            
        try:
            # Obtiene las amplitudes complejas del estado cuántico
            amplitudes = quantum_circuit.get_complex_amplitudes()
            
            # Validación básica de las amplitudes
            if not amplitudes or len(amplitudes) == 0:
                raise ValueError("El circuito no tiene amplitudes válidas")
                
            return self.process_quantum_state(amplitudes)
            
        except Exception as e:
            logger.exception(f"Error al procesar el circuito cuántico: {str(e)}")
            raise

    def process_quantum_state(self, quantum_state: QuantumState) -> ProcessedResult:
        """
        Procesa un estado cuántico aplicando la FFT y extrayendo características frecuenciales.
        
        Este método realiza las siguientes operaciones:
        1. Aplica la Transformada de Fourier al estado cuántico
        2. Calcula magnitudes y fases del espectro
        3. Determina la entropía de Shannon de las magnitudes
        4. Calcula una medida de coherencia basada en la varianza de fase
        
        Args:
            quantum_state: Lista de amplitudes complejas que representan un estado cuántico.
            
        Returns:
            Diccionario con los resultados del análisis:
            - 'magnitudes': Array de magnitudes del espectro FFT
            - 'phases': Array de fases del espectro FFT
            - 'entropy': Valor de entropía de Shannon de las magnitudes
            - 'coherence': Medida de coherencia derivada de la varianza de fase
            - 'dominant_frequencies': Índices de las frecuencias dominantes
            
        Raises:
            ValueError: Si el estado cuántico está vacío.
            TypeError: Si el estado cuántico contiene valores no compatibles.
        """
        if not quantum_state:
            msg = "El estado cuántico no puede estar vacío."
            logger.error(msg)
            raise ValueError(msg)

        # Usa el mecanismo de caché para obtener o calcular el resultado
        return self._cached_compute(
            tuple(quantum_state),  # Clave de caché: tupla de valores complejos
            self._process_quantum_state_internal,  # Función de cálculo real
            quantum_state  # Argumentos para la función
        )

    def _process_quantum_state_internal(self, quantum_state: QuantumState) -> ProcessedResult:
        """
        Implementación interna del procesamiento FFT del estado cuántico.
        
        Args:
            quantum_state: Lista de amplitudes complejas.
            
        Returns:
            Diccionario con los resultados del análisis.
        """
        try:
            # Convierte la lista en un array de complejos
            quantum_state_array = np.array(quantum_state, dtype=complex)
        except Exception as e:
            logger.exception("Error al convertir el estado cuántico a np.array")
            raise TypeError(f"Estado cuántico inválido: {str(e)}") from e

        # Aplica la FFT al estado cuántico
        fft_result = np.fft.fft(quantum_state_array)
        
        # Calcula magnitudes y fases
        fft_magnitudes = np.abs(fft_result)
        fft_phases = np.angle(fft_result)
        
        # Normalización de magnitudes (suma = 1)
        norm_magnitudes = fft_magnitudes / np.sum(fft_magnitudes)
        
        # Análisis estadístico
        entropy = self.stat_analysis.shannon_entropy(fft_magnitudes.tolist())
        phase_variance = np.var(fft_phases)
        coherence = np.exp(-phase_variance)
        
        # Identifica frecuencias dominantes (aquellas con magnitud > media + desviación)
        threshold = np.mean(fft_magnitudes) + np.std(fft_magnitudes)
        dominant_freqs = np.where(fft_magnitudes > threshold)[0]
        
        # Construye y devuelve el resultado
        result = {
            'magnitudes': fft_magnitudes,
            'norm_magnitudes': norm_magnitudes,
            'phases': fft_phases,
            'entropy': entropy,
            'coherence': coherence,
            'phase_variance': phase_variance,
            'dominant_frequencies': dominant_freqs
        }
        
        logger.debug(f"Estado cuántico procesado: entropía={entropy:.4f}, coherencia={coherence:.4f}")
        return result

    def fft_based_initializer(self, quantum_state: QuantumState, out_dimension: int, 
                             scale: float = 0.01) -> torch.Tensor:
        """
        Inicializa una matriz de pesos basada en la FFT del estado cuántico.
        
        Este inicializador crea una matriz de pesos donde cada fila es una réplica
        de las magnitudes normalizadas de la FFT, escaladas por un factor.
        
        Args:
            quantum_state: Lista de amplitudes complejas del estado cuántico.
            out_dimension: Número de filas (dimensión de salida) de la matriz.
            scale: Factor de escala para los valores de la matriz.
            
        Returns:
            Tensor de PyTorch con la matriz de pesos inicializada.
            
        Raises:
            ValueError: Si los parámetros de dimensión son inválidos.
        """
        self._validate_dimensions(len(quantum_state), out_dimension)
        self._validate_scale(scale)
        
        # Procesa el estado para obtener las magnitudes
        processed = self.process_quantum_state(quantum_state)
        norm_magnitudes = processed['norm_magnitudes']
        
        # Crea la matriz replicando el vector de magnitudes normalizadas
        weight_matrix = scale * np.tile(norm_magnitudes, (out_dimension, 1))
        
        # Asegura que la matriz tenga la forma correcta
        if weight_matrix.shape != (out_dimension, len(quantum_state)):
            logger.warning(f"La forma de la matriz ({weight_matrix.shape}) no coincide con las dimensiones esperadas "
                          f"({out_dimension}, {len(quantum_state)})")
        
        return torch.tensor(weight_matrix, dtype=torch.float32)

    def advanced_fft_initializer(self, quantum_state: QuantumState, out_dimension: int, 
                               in_dimension: Optional[int] = None, scale: float = 0.01, 
                               use_phases: bool = True, randomize: bool = False) -> torch.Tensor:
        """
        Inicializador avanzado que crea una matriz rectangular utilizando magnitudes y fases de la FFT.
        
        Este inicializador crea una matriz de pesos más sofisticada que incorpora:
        - Magnitudes normalizadas de la FFT
        - Opcionalmente información de fase
        - Desplazamientos para crear diversidad en las filas
        - Opcionalmente aleatorización para romper simetrías
        
        Args:
            quantum_state: Lista de amplitudes complejas del estado cuántico.
            out_dimension: Número de filas (dimensión de salida) de la matriz.
            in_dimension: Número de columnas. Si es None, se usa la longitud del estado.
            scale: Factor de escala para los valores de la matriz.
            use_phases: Si es True, modula magnitudes por fases.
            randomize: Si es True, añade un pequeño ruido aleatorio.
            
        Returns:
            Tensor de PyTorch con la matriz de pesos inicializada.
            
        Raises:
            ValueError: Si los parámetros de dimensión son inválidos.
        """
        # Define la dimensión de entrada si no se especifica
        in_dim = in_dimension if in_dimension is not None else len(quantum_state)
        
        # Validaciones
        self._validate_dimensions(len(quantum_state), out_dimension, in_dim)
        self._validate_scale(scale)
        
        # Procesa el estado para obtener magnitudes y fases
        processed = self.process_quantum_state(quantum_state)
        norm_magnitudes = processed['norm_magnitudes']
        phases = processed['phases']
        
        # Construye el vector base para la matriz
        base_features = self._prepare_base_features(norm_magnitudes, in_dim)
        
        # Incorpora información de fase si se solicita
        if use_phases:
            phase_features = self._prepare_base_features(phases, in_dim)
            # Modula base_features con la información de fase
            base_features = base_features * (1 + 0.1 * np.cos(phase_features))
        
        # Crea la matriz de pesos desplazando el vector base para cada fila
        weight_matrix = self._create_weight_matrix(base_features, out_dimension, in_dim)
        
        # Añade aleatorización si se solicita
        if randomize:
            weight_matrix += np.random.normal(0, scale * 0.1, weight_matrix.shape)
        
        # Escala y normaliza la matriz
        weight_matrix = scale * weight_matrix / np.max(np.abs(weight_matrix) + 1e-8)
        
        return torch.tensor(weight_matrix, dtype=torch.float32)

    def spectral_initializer(self, quantum_state: QuantumState, out_dimension: int, 
                           in_dimension: Optional[int] = None, scale: float = 0.01,
                           spectral_filtering: bool = True) -> torch.Tensor:
        """
        Inicializador basado en propiedades espectrales del estado cuántico.
        
        Este inicializador más avanzado incorpora:
        - Análisis espectral completo (magnitud y fase)
        - Filtrado espectral opcional para priorizar frecuencias dominantes
        - Modelado de correlaciones espectrales
        
        Args:
            quantum_state: Lista de amplitudes complejas del estado cuántico.
            out_dimension: Número de filas (dimensión de salida) de la matriz.
            in_dimension: Número de columnas. Si es None, se usa la longitud del estado.
            scale: Factor de escala para los valores de la matriz.
            spectral_filtering: Si es True, enfatiza las frecuencias dominantes.
            
        Returns:
            Tensor de PyTorch con la matriz de pesos inicializada.
        """
        # Define la dimensión de entrada si no se especifica
        in_dim = in_dimension if in_dimension is not None else len(quantum_state)
        
        # Validaciones
        self._validate_dimensions(len(quantum_state), out_dimension, in_dim)
        self._validate_scale(scale)
        
        # Procesa el estado para obtener características espectrales
        processed = self.process_quantum_state(quantum_state)
        fft_magnitudes = processed['magnitudes']
        fft_phases = processed['phases']
        dominant_freqs = processed['dominant_frequencies']
        
        # Crear una representación espectral completa
        spectral_representation = fft_magnitudes * np.exp(1j * fft_phases)
        
        # Filtrado espectral - enfatiza frecuencias dominantes si se solicita
        if spectral_filtering and len(dominant_freqs) > 0:
            enhancement = np.ones(len(fft_magnitudes))
            enhancement[dominant_freqs] = 2.0  # Duplica la importancia de frecuencias dominantes
            spectral_representation *= enhancement
        
        # Genera una matriz de correlación espectral
        n_freqs = len(spectral_representation)
        corr_matrix = np.zeros((n_freqs, n_freqs), dtype=complex)
        
        for i in range(n_freqs):
            for j in range(n_freqs):
                # Modelo simple de correlación entre frecuencias
                phase_diff = np.abs(fft_phases[i] - fft_phases[j])
                mag_product = np.sqrt(fft_magnitudes[i] * fft_magnitudes[j])
                corr_matrix[i, j] = mag_product * np.exp(-0.5 * phase_diff)
        
        # Extrae los valores propios y vectores propios principales
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.abs(corr_matrix))
            # Ordena por valor propio descendente
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except np.linalg.LinAlgError:
            # Fallback en caso de error
            logger.warning("Error en descomposición de valores propios, usando método alternativo")
            eigenvectors = np.eye(n_freqs)
            eigenvalues = np.ones(n_freqs)
            
        # Usa los vectores propios principales para construir la matriz de pesos
        n_components = min(out_dimension, len(eigenvalues))
        
        # Construye la matriz inicial con los vectores propios principales
        weight_basis = eigenvectors[:, :n_components].T
        
        # Adapta dimensiones si es necesario
        if in_dim != n_freqs or out_dimension != n_components:
            weight_matrix = np.zeros((out_dimension, in_dim))
            for i in range(min(out_dimension, n_components)):
                # Repetir o truncar según sea necesario
                for j in range(in_dim):
                    weight_matrix[i, j] = weight_basis[i % n_components, j % n_freqs]
        else:
            weight_matrix = weight_basis
        
        # Escala y normaliza
        weight_matrix = scale * weight_matrix / (np.max(np.abs(weight_matrix)) + 1e-8)
        
        return torch.tensor(weight_matrix.real, dtype=torch.float32)

    def visualize_fft_analysis(self, quantum_state: QuantumState, 
                              title: str = "Análisis FFT de Estado Cuántico") -> None:
        """
        Visualiza los resultados del análisis FFT del estado cuántico.
        
        Args:
            quantum_state: Lista de amplitudes complejas del estado cuántico.
            title: Título para la visualización.
        """
        if not quantum_state:
            raise ValueError("El estado cuántico no puede estar vacío")
            
        # Procesa el estado cuántico
        result = self.process_quantum_state(quantum_state)
        
        # Configura la visualización
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # 1. Magnitudes de la FFT
        ax1 = axes[0, 0]
        x = np.arange(len(result['magnitudes']))
        ax1.bar(x, result['magnitudes'], alpha=0.7, width=0.8)
        ax1.set_title("Magnitudes FFT")
        ax1.set_xlabel("Frecuencia")
        ax1.set_ylabel("Magnitud")
        ax1.grid(alpha=0.3)
        
        # Resaltar frecuencias dominantes
        dominant_freqs = result['dominant_frequencies']
        if len(dominant_freqs) > 0:
            ax1.bar(dominant_freqs, result['magnitudes'][dominant_freqs], 
                   color='red', alpha=0.7, width=0.8)
            for freq in dominant_freqs:
                ax1.text(freq, result['magnitudes'][freq], f"{freq}", 
                        ha='center', va='bottom', fontsize=8, rotation=45)
        
        # 2. Fases de la FFT
        ax2 = axes[0, 1]
        ax2.stem(x, result['phases'], basefmt=" ")
        ax2.set_title("Fases FFT")
        ax2.set_xlabel("Frecuencia")
        ax2.set_ylabel("Fase (radianes)")
        ax2.grid(alpha=0.3)
        
        # 3. Estado cuántico original (módulo al cuadrado - probabilidades)
        ax3 = axes[1, 0]
        probabilities = np.abs(np.array(quantum_state))**2
        ax3.bar(np.arange(len(quantum_state)), probabilities, alpha=0.7, width=0.8)
        ax3.set_title("Probabilidades del Estado Cuántico")
        ax3.set_xlabel("Estado")
        ax3.set_ylabel("Probabilidad")
        ax3.grid(alpha=0.3)
        
        # 4. Información de entropía y coherencia
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = (
            f"Entropía: {result['entropy']:.4f}\n"
            f"Coherencia: {result['coherence']:.4f}\n"
            f"Varianza de fase: {result['phase_variance']:.4f}\n"
            f"Número de frecuencias dominantes: {len(result['dominant_frequencies'])}\n"
            f"Frecuencias dominantes: {', '.join(map(str, result['dominant_frequencies']))}\n\n"
            
            f"Interpretación:\n"
            f"- Entropía alta (>{0.7:.1f}) indica mayor aleatoriedad\n"
            f"- Coherencia alta (>{0.6:.1f}) indica mayor orden\n"
            f"- Frecuencias dominantes representan patrones clave"
        )
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
    def visualize_initializer(self, quantum_state: QuantumState, out_dimension: int, 
                             in_dimension: Optional[int] = None,
                             initializer_type: str = 'advanced') -> None:
        """
        Visualiza la matriz de pesos generada por diferentes inicializadores.
        
        Args:
            quantum_state: Estado cuántico de entrada.
            out_dimension: Dimensión de salida.
            in_dimension: Dimensión de entrada (opcional).
            initializer_type: Tipo de inicializador ('basic', 'advanced' o 'spectral').
        """
        in_dim = in_dimension if in_dimension is not None else len(quantum_state)
        
        # Genera la matriz según el tipo especificado
        if initializer_type == 'basic':
            matrix = self.fft_based_initializer(quantum_state, out_dimension)
            method_name = "Inicializador FFT Básico"
        elif initializer_type == 'spectral':
            matrix = self.spectral_initializer(quantum_state, out_dimension, in_dim)
            method_name = "Inicializador Espectral"
        else:  # 'advanced' por defecto
            matrix = self.advanced_fft_initializer(quantum_state, out_dimension, in_dim)
            method_name = "Inicializador FFT Avanzado"
        
        # Convierte el tensor a numpy para visualización
        weight_matrix = matrix.numpy()
        
        # Visualización
        plt.figure(figsize=(10, 8))
        plt.imshow(weight_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Valor de peso')
        plt.title(f"{method_name} ({weight_matrix.shape[0]}x{weight_matrix.shape[1]})")
        plt.xlabel("Dimensión de entrada")
        plt.ylabel("Dimensión de salida")
        
        # Añadir estadísticas
        stats_text = (
            f"Media: {np.mean(weight_matrix):.4f}\n"
            f"Desv. Est.: {np.std(weight_matrix):.4f}\n"
            f"Min: {np.min(weight_matrix):.4f}\n"
            f"Max: {np.max(weight_matrix):.4f}"
        )
        
        plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

    def _prepare_base_features(self, feature_vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Prepara un vector de características con la dimensión deseada.
        
        Args:
            feature_vector: Vector de características original.
            target_dim: Dimensión objetivo.
            
        Returns:
            Vector de características redimensionado.
        """
        if len(feature_vector) >= target_dim:
            # Truncar si el original es más grande
            return feature_vector[:target_dim]
        else:
            # Repetir si el original es más pequeño
            repeats = int(np.ceil(target_dim / len(feature_vector)))
            return np.tile(feature_vector, repeats)[:target_dim]

    def _create_weight_matrix(self, base_features: np.ndarray, out_dimension: int, in_dimension: int) -> np.ndarray:
        """
        Crea una matriz de pesos desplazando circularmente un vector base.
        
        Args:
            base_features: Vector base para construir la matriz.
            out_dimension: Número de filas.
            in_dimension: Número de columnas.
            
        Returns:
            Matriz de pesos generada.
        """
        if len(base_features) != in_dimension:
            raise ValueError(f"Dimensión del vector base ({len(base_features)}) no coincide "
                          f"con dimensión de entrada ({in_dimension})")
            
        weight_matrix = np.empty((out_dimension, in_dimension))
        
        for i in range(out_dimension):
            # Calcular el desplazamiento para esta fila
            shift = i % in_dimension
            # Aplicar desplazamiento circular al vector base
            weight_matrix[i] = np.roll(base_features, shift)
            
        return weight_matrix

    def _validate_dimensions(self, state_len: int, out_dimension: int, in_dimension: Optional[int] = None) -> None:
        """
        Valida las dimensiones para los inicializadores.
        
        Args:
            state_len: Longitud del estado cuántico.
            out_dimension: Dimensión de salida.
            in_dimension: Dimensión de entrada (opcional).
            
        Raises:
            ValueError: Si las dimensiones son inválidas.
        """
        if out_dimension <= 0:
            raise ValueError(f"La dimensión de salida debe ser positiva, recibido: {out_dimension}")
            
        if in_dimension is not None and in_dimension <= 0:
            raise ValueError(f"La dimensión de entrada debe ser positiva, recibido: {in_dimension}")
            
        if state_len <= 0:
            raise ValueError("El estado cuántico debe tener al menos un elemento")

    def _validate_scale(self, scale: float) -> None:
        """
        Valida el factor de escala para los inicializadores.
        
        Args:
            scale: Factor de escala a validar.
            
        Raises:
            ValueError: Si el factor de escala es inválido.
        """
        if scale <= 0:
            raise ValueError(f"El factor de escala debe ser positivo, recibido: {scale}")

# Ejemplo de uso 
if __name__ == "__main__":
    # 1. Crear un estado cuántico de ejemplo
    example_state = [complex(0.5, 0), complex(0, 0.5), complex(0.5, 0), complex(0.5, 0)]
    
    # 2. Inicializar el integrador
    integrator = FFTBayesIntegrator(cache_size=100)
    
    # 3. Procesar el estado cuántico y mostrar resultados
    try:
        result = integrator.process_quantum_state(example_state)
        print("\nResultados del procesamiento:")
        print(f"Entropía: {result['entropy']:.4f}")
        print(f"Coherencia: {result['coherence']:.4f}")
        print(f"Frecuencias dominantes: {result['dominant_frequencies']}")
        
        # 4. Visualizar el análisis FFT
        integrator.visualize_fft_analysis(example_state, "Análisis FFT del Estado de Ejemplo")
        
        # 5. Probar diferentes inicializadores
        # Dimensiones para las matrices de peso
        out_dim = 8
        in_dim = 6
        
        # Inicializador básico
        basic_weights = integrator.fft_based_initializer(
            quantum_state=example_state,
            out_dimension=out_dim
        )
        print("\nForma de matriz con inicializador básico:", basic_weights.shape)
        
        # Inicializador avanzado
        advanced_weights = integrator.advanced_fft_initializer(
            quantum_state=example_state,
            out_dimension=out_dim,
            in_dimension=in_dim,
            use_phases=True,
            randomize=True
        )
        print("Forma de matriz con inicializador avanzado:", advanced_weights.shape)
        
        # Inicializador espectral
        spectral_weights = integrator.spectral_initializer(
            quantum_state=example_state,
            out_dimension=out_dim,
            in_dimension=in_dim,
            spectral_filtering=True
        )
        print("Forma de matriz con inicializador espectral:", spectral_weights.shape)
        
        # 6. Visualizar las matrices generadas
        print("\nVisualizando matrices de peso...")
        integrator.visualize_initializer(example_state, out_dim, in_dim, 'basic')
        integrator.visualize_initializer(example_state, out_dim, in_dim, 'advanced')
        integrator.visualize_initializer(example_state, out_dim, in_dim, 'spectral')
        
        # 7. Mostrar estadísticas de caché
        cache_stats = integrator.get_cache_stats()
        print("\nEstadísticas de caché:")
        for key, value in cache_stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        
    finally:
        # 8. Limpiar caché
        integrator.clear_cache()
        print("\nCaché limpiada")

    # 9. Ejemplo con un circuito cuántico (asumiendo que existe la clase)
    try:
        circuit = ResilientQuantumCircuit()  # Asumiendo que existe esta clase
        circuit_result = integrator.process_quantum_circuit(circuit)
        print("\nResultados del circuito cuántico procesado:")
        print(f"Entropía: {circuit_result['entropy']:.4f}")
        print(f"Coherencia: {circuit_result['coherence']:.4f}")
        
    except Exception as e:
        print(f"Error al procesar circuito: {str(e)}")

