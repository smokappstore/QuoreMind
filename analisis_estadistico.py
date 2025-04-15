import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Tuple, Union, Optional, Dict, Any
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import logging
import matplotlib.pyplot as plt

# Configuración del log
logger = logging.getLogger(__name__)

class StatisticalAnalysis:
    """
    Clase para realizar análisis estadísticos avanzados con datos multidimensionales.
    
    Esta clase proporciona métodos para calcular medidas estadísticas como la entropía,
    cosenos direccionales, matrices de covarianza y distancias estadísticas. Permite
    evaluar características de distribuciones de datos y relaciones entre variables.
    
    Métodos principales:
      - shannon_entropy: Calcula la entropía de Shannon para medir incertidumbre.
      - calculate_cosines: Determina los cosenos direccionales en un espacio 3D.
      - calculate_covariance_matrix: Computa la matriz de covarianza para datos tensoriales.
      - compute_mahalanobis_distance: Calcula la distancia de Mahalanobis para detectar outliers.
      - visualize_entropy: Visualiza la entropía en forma gráfica.
      - visualize_cosines: Visualiza la dirección de los vectores mediante cosenos.
    
    Ejemplo de uso:
        >>> datos = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
        >>> entropy = StatisticalAnalysis.shannon_entropy(datos)
        >>> print(f"Entropía: {entropy}")
    """
    
    @staticmethod
    def shannon_entropy(data: List[float], decimals: int = 6) -> float:
        """
        Calcula la entropía de Shannon de un conjunto de datos.
        
        La entropía de Shannon mide la cantidad de información o incertidumbre
        en una distribución de probabilidad. Un valor mayor indica mayor
        incertidumbre o aleatoriedad en los datos.
        
        Fórmula: H(X) = -∑(p(x) * log₂(p(x)))
        
        Args:
            data: Lista de valores numéricos.
            decimals: Número de decimales para redondeo antes de calcular probabilidades.
                      Valores más altos preservan más precisión pero pueden reducir agrupamiento.
        
        Returns:
            Valor de entropía calculado.
            
        Raises:
            ValueError: Si la lista de datos está vacía.
        """
        if not data:
            raise ValueError("La lista de datos no puede estar vacía")
        
        # Verificar que todos los elementos son numéricos
        if not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("Todos los elementos de la lista deben ser numéricos")
            
        # Redondeo para evitar problemas con pequeñas diferencias numéricas
        rounded_data = np.round(data, decimals=decimals)
        
        # Cálculo de frecuencias y probabilidades
        values, counts = np.unique(rounded_data, return_counts=True)
        probabilities = counts / len(data)
        
        # Filtrar probabilidades para evitar log(0)
        nonzero_probs = probabilities[probabilities > 0]
        
        # Cálculo de la entropía
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        
        logger.debug(f"Entropía calculada: {entropy} (con redondeo a {decimals} decimales)")
        return entropy
    
    @staticmethod
    def calculate_cosines(entropy: float, prn_object: float, epsilon: float = 1e-6) -> Tuple[float, float, float]:
        """
        Calcula los cosenos direccionales en un espacio tridimensional.
        
        Este método determina los cosenos de los ángulos formados por un vector
        en el espacio 3D con cada uno de los ejes coordenados. Estos cosenos
        describen la dirección del vector con origen (0,0,0) y punto final
        (entropy, prn_object, 1).
        
        Args:
            entropy: Componente X del vector (entropía).
            prn_object: Componente Y del vector (valor PRN).
            epsilon: Valor mínimo para evitar divisiones por cero.
        
        Returns:
            Tupla con los tres cosenos direccionales (cos_x, cos_y, cos_z).
        """
        # Validación de entradas
        StatisticalAnalysis._validate_numeric(entropy, "entropy")
        StatisticalAnalysis._validate_numeric(prn_object, "prn_object")
        
        # Evitar divisiones por cero
        safe_entropy = max(abs(entropy), epsilon)
        safe_prn = max(abs(prn_object), epsilon)
        
        # Preservar signos originales
        safe_entropy = safe_entropy * (1 if entropy >= 0 else -1)
        safe_prn = safe_prn * (1 if prn_object >= 0 else -1)
        
        # Cálculo de la magnitud del vector
        magnitude = np.sqrt(safe_entropy**2 + safe_prn**2 + 1)
        
        # Cálculo de los cosenos direccionales
        cos_x = safe_entropy / magnitude
        cos_y = safe_prn / magnitude
        cos_z = 1 / magnitude
        
        logger.debug(f"Cosenos direccionales calculados: X={cos_x:.4f}, Y={cos_y:.4f}, Z={cos_z:.4f}")
        return cos_x, cos_y, cos_z
    
    @staticmethod
    def calculate_covariance_matrix(data: tf.Tensor, sample_axis: int = 0) -> np.ndarray:
        """
        Calcula la matriz de covarianza a partir de un tensor de datos de TensorFlow.
        
        La matriz de covarianza es una matriz cuadrada que describe la variabilidad
        conjunta de múltiples variables. Cada elemento (i,j) representa la covarianza
        entre las variables i y j.
        
        Args:
            data: Tensor de TensorFlow con los datos.
            sample_axis: Eje que representa las muestras (observaciones).
        
        Returns:
            Matriz de covarianza como array de NumPy.
            
        Raises:
            ValueError: Si el tensor está vacío o tiene dimensiones incorrectas.
        """
        if data is None or tf.size(data) == 0:
            raise ValueError("El tensor de datos no puede estar vacío")
        
        try:
            # Verificar que el tensor tiene al menos 2 dimensiones
            if len(data.shape) < 2:
                data = tf.expand_dims(data, axis=-1)
                logger.warning("Tensor expandido a 2 dimensiones para cálculo de covarianza")
            
            # Cálculo de la matriz de covarianza
            cov_matrix = tfp.stats.covariance(data, sample_axis=sample_axis, event_axis=None)
            result = cov_matrix.numpy()
            
            # Verificación de validez de la matriz (debe ser simétrica)
            if not np.allclose(result, result.T, rtol=1e-5, atol=1e-8):
                logger.warning("La matriz de covarianza calculada no es perfectamente simétrica")
                # Forzar simetría para estabilidad numérica
                result = (result + result.T) / 2
                
            return result
            
        except Exception as e:
            logger.error(f"Error al calcular la matriz de covarianza: {str(e)}")
            raise
    
    @staticmethod
    def compute_mahalanobis_distance(data: List[List[float]], point: List[float], 
                                    regularization: float = 1e-6) -> float:
        """
        Calcula la distancia de Mahalanobis de un punto respecto a un conjunto de datos.
        
        La distancia de Mahalanobis mide la distancia entre un punto y una distribución,
        teniendo en cuenta la estructura de covarianza de los datos. Es útil para detectar
        valores atípicos (outliers) en datos multivariantes.
        
        Args:
            data: Lista de listas donde cada lista interna representa un punto multivariante.
            point: Lista que representa el punto cuya distancia se quiere calcular.
            regularization: Factor de regularización para estabilidad numérica.
        
        Returns:
            Distancia de Mahalanobis calculada.
            
        Raises:
            ValueError: Si los datos o el punto tienen dimensiones incompatibles.
        """
        # Validación de entradas
        if not data or not point:
            raise ValueError("Los datos y el punto no pueden estar vacíos")
        
        if len(point) != len(data[0]):
            raise ValueError(f"Dimensiones incompatibles: datos ({len(data[0])}D) vs punto ({len(point)}D)")
        
        # Conversión a arrays de NumPy
        data_array = np.array(data)
        point_array = np.array(point)
        
        try:
            # Cálculo de la matriz de covarianza usando scikit-learn
            covariance_estimator = EmpiricalCovariance(assume_centered=False)
            covariance_estimator.fit(data_array)
            cov_matrix = covariance_estimator.covariance_
            
            # Regularización para estabilidad numérica
            if regularization > 0:
                cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
                logger.debug(f"Matriz de covarianza regularizada con factor {regularization}")
            
            # Cálculo de la inversa de la matriz de covarianza
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError as e:
                logger.warning(f"Error en inversión estándar, usando pseudo-inversa: {str(e)}")
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
            
            # Cálculo del vector de medias
            mean_vector = np.mean(data_array, axis=0)
            
            # Cálculo de la distancia de Mahalanobis
            distance = mahalanobis(point_array, mean_vector, inv_cov_matrix)
            
            logger.debug(f"Distancia de Mahalanobis calculada: {distance:.4f}")
            return distance
            
        except Exception as e:
            logger.error(f"Error en el cálculo de la distancia de Mahalanobis: {str(e)}")
            raise
    
    @staticmethod
    def visualize_entropy(data: List[float], bins: int = 10, title: str = "Distribución y Entropía") -> None:
        """
        Visualiza la distribución de los datos y su entropía.
        
        Args:
            data: Lista de valores numéricos.
            bins: Número de bins para el histograma.
            title: Título para la gráfica.
        """
        if not data:
            raise ValueError("La lista de datos no puede estar vacía")
        
        entropy = StatisticalAnalysis.shannon_entropy(data)
        
        plt.figure(figsize=(10, 6))
        
        # Crear histograma
        n, bins_edges, patches = plt.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue')
        
        # Añadir línea de densidad de kernel
        from scipy import stats
        x = np.linspace(min(data), max(data), 1000)
        kde = stats.gaussian_kde(data)
        plt.plot(x, kde(x), 'r-', linewidth=2, label='Densidad KDE')
        
        # Añadir información de entropía
        plt.text(0.7, 0.85, f'Entropía: {entropy:.4f}', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Añadir umbral de alta entropía visual
        high_entropy = 0.8 * np.log2(bins)  # Umbral relativo al máximo teórico
        plt.axhline(y=high_entropy, color='g', linestyle='--', alpha=0.6,
                   label=f'Umbral Alta Entropía: {high_entropy:.4f}')
        
        plt.title(title)
        plt.xlabel('Valor')
        plt.ylabel('Densidad de probabilidad')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_cosines(cosines: Tuple[float, float, float]) -> None:
        """
        Visualiza los cosenos direccionales como un vector en un espacio 3D.
        
        Args:
            cosines: Tupla con los tres cosenos direccionales (cos_x, cos_y, cos_z).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Obtener componentes
        cos_x, cos_y, cos_z = cosines
        
        # Dibujar ejes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, label='Eje X')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, label='Eje Y')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, label='Eje Z')
        
        # Dibujar vector de dirección
        ax.quiver(0, 0, 0, cos_x, cos_y, cos_z, color='purple', arrow_length_ratio=0.15, 
                 linewidth=2, label='Vector de dirección')
        
        # Configuración de la visualización
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Visualización de Cosenos Direccionales')
        
        # Añadir texto con valores
        text_str = f'cos_x: {cos_x:.4f}\ncos_y: {cos_y:.4f}\ncos_z: {cos_z:.4f}'
        ax.text(0.7, 0.7, 0.7, text_str, bbox=dict(facecolor='white', alpha=0.7))
        
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_mahalanobis(data: List[List[float]], test_points: List[List[float]], 
                             threshold: float = 3.0) -> None:
        """
        Visualiza la distancia de Mahalanobis para múltiples puntos de prueba.
        
        Args:
            data: Lista de listas donde cada lista interna representa un punto multivariante.
            test_points: Lista de puntos a evaluar.
            threshold: Umbral para considerar un punto como atípico (outlier).
        """
        if len(data[0]) != 2:
            logger.warning("La visualización de Mahalanobis solo está implementada para datos 2D")
            return
        
        data_array = np.array(data)
        test_points_array = np.array(test_points)
        
        # Calcular distancias de Mahalanobis
        distances = []
        for point in test_points:
            dist = StatisticalAnalysis.compute_mahalanobis_distance(data, point)
            distances.append(dist)
        
        # Visualización
        plt.figure(figsize=(10, 8))
        
        # Graficar datos originales
        plt.scatter(data_array[:, 0], data_array[:, 1], c='blue', alpha=0.5, label='Datos de referencia')
        
        # Graficar puntos de prueba con color según distancia
        sc = plt.scatter(test_points_array[:, 0], test_points_array[:, 1], 
                        c=distances, cmap='YlOrRd', s=100, 
                        norm=plt.Normalize(0, max(threshold * 1.5, max(distances))),
                        edgecolors='black', label='Puntos de prueba')
        
        # Añadir barra de colores
        cbar = plt.colorbar(sc)
        cbar.set_label('Distancia de Mahalanobis')
        
        # Marcar umbral
        cbar.ax.axhline(y=threshold, color='r', linestyle='--')
        cbar.ax.text(0.5, threshold, f' Umbral ({threshold})', va='center', ha='left', color='red')
        
        # Identificar outliers
        for i, (point, dist) in enumerate(zip(test_points, distances)):
            if dist > threshold:
                plt.annotate(f'Outlier {i}\n({dist:.2f})', (point[0], point[1]), 
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        # Dibujar elipse de confianza (para datos 2D)
        from matplotlib.patches import Ellipse
        cov = EmpiricalCovariance().fit(data_array).covariance_
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor='none', edgecolor='green', linestyle='--')
        
        # Ubicar elipse en posición y escala correcta
        mean = np.mean(data_array, axis=0)
        eigval, eigvec = np.linalg.eigh(cov)
        sqrt_eigval = np.sqrt(eigval)
        
        transform = (
            plt.gca().transData +
            plt.matplotlib.transforms.Affine2D().rotate_deg(np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))) +
            plt.matplotlib.transforms.Affine2D().scale(threshold * sqrt_eigval[0], threshold * sqrt_eigval[1]) +
            plt.matplotlib.transforms.Affine2D().translate(*mean)
        )
        
        ellipse.set_transform(transform)
        plt.gca().add_patch(ellipse)
        
        plt.title('Visualización de Distancia de Mahalanobis')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _validate_numeric(value: Any, parameter_name: str) -> None:
        """
        Valida que un valor sea numérico.
        
        Args:
            value: Valor a validar.
            parameter_name: Nombre del parámetro para mensajes de error.
            
        Raises:
            TypeError: Si el valor no es numérico.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"El parámetro {parameter_name} debe ser numérico, recibido: {type(value)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de cálculo de entropía
    datos_entropia = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3]
    entropia = StatisticalAnalysis.shannon_entropy(datos_entropia)
    print(f"Entropía calculada: {entropia}")
    
    # Visualizar la entropía
    StatisticalAnalysis.visualize_entropy(datos_entropia)
    
    # Ejemplo de cálculo y visualización de cosenos direccionales
    cosenos = StatisticalAnalysis.calculate_cosines(0.7, 0.4)
    print(f"Cosenos direccionales: {cosenos}")
    StatisticalAnalysis.visualize_cosines(cosenos)
    
    # Ejemplo de cálculo de distancia de Mahalanobis
    datos_2d = [[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [2, 2]]
    puntos_prueba = [[2, 2], [5, 5], [0, 0], [3, 3]]
    
    for i, punto in enumerate(puntos_prueba):
        distancia = StatisticalAnalysis.compute_mahalanobis_distance(datos_2d, punto)
        print(f"Distancia de Mahalanobis para punto {i}: {distancia}")
    
    # Visualizar distancias de Mahalanobis
    StatisticalAnalysis.visualize_mahalanobis(datos_2d, puntos_prueba)
