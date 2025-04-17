import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

class NodoFotonico:
    """
    Simula un nodo fotónico que procesa información cuántica
    y aplica inferencia bayesiana para la toma de decisiones.
    """
    def __init__(self, id, dim=2):
        self.id = id
        self.dimension = dim
        self.estado = np.random.rand(dim) + 1j * np.random.rand(dim)
        self.estado = self.estado / np.linalg.norm(self.estado)
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
    
    def actualizar_estado(self, input_estado):
        """Actualiza el estado del nodo basado en entrada y coherencia"""
        # Simula interferencia cuántica
        nuevo_estado = self.estado + 0.5 * input_estado
        self.estado = nuevo_estado / np.linalg.norm(nuevo_estado)
        return self.estado
    
    def inferencia_bayesiana(self, observacion):
        """Actualiza conocimiento usando inferencia bayesiana"""
        # Actualiza parámetros de distribución beta (modelo bayesiano simple)
        if observacion == 1:
            self.prior_alpha += 1
        else:
            self.prior_beta += 1
        
        # Calcula probabilidad posterior
        prob = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        return prob
    
    def crear_circuito_cuantico(self):
        """Crea un circuito cuántico simple basado en el estado actual"""
        qc = QuantumCircuit(self.dimension)
        
        # Aplicamos puertas basadas en el estado actual
        for i in range(self.dimension):
            # Convertimos amplitudes complejas a ángulos para rotaciones
            theta = np.angle(self.estado[i])
            qc.rx(theta, i)
            
        return qc


class RedNodosFotonicos:
    """Implementa una red de nodos fotónicos para procesamiento distribuido"""
    def __init__(self, num_nodos=3, dimension=2):
        self.nodos = [NodoFotonico(i, dimension) for i in range(num_nodos)]
        self.matriz_conexiones = np.random.rand(num_nodos, num_nodos)
        np.fill_diagonal(self.matriz_conexiones, 0)  # No auto-conexiones
    
    def propagar_informacion(self, input_datos):
        """Propaga información a través de la red de nodos"""
        estados_resultantes = []
        
        # Inicializar estados de entrada
        for i, nodo in enumerate(self.nodos):
            if i < len(input_datos):
                # Codificar entrada en estado cuántico
                estado_entrada = np.array(input_datos[i])
                if estado_entrada.ndim == 0:  # Si es un escalar
                    estado_entrada = np.array([np.sqrt(1-estado_entrada**2), estado_entrada])
                estados_resultantes.append(nodo.actualizar_estado(estado_entrada))
            else:
                estados_resultantes.append(nodo.estado)
        
        # Simular comunicación entre nodos (teleportación cuántica simplificada)
        for _ in range(3):  # Iteraciones de propagación
            nuevos_estados = []
            for i, nodo in enumerate(self.nodos):
                # Calcula influencia ponderada de otros nodos
                estado_combinado = np.zeros(nodo.dimension, dtype=complex)
                for j, otro_nodo in enumerate(self.nodos):
                    if i != j:
                        peso = self.matriz_conexiones[i, j]
                        estado_combinado += peso * estados_resultantes[j]
                
                # Normalizar y actualizar
                if np.linalg.norm(estado_combinado) > 0:
                    estado_combinado = estado_combinado / np.linalg.norm(estado_combinado)
                    nuevos_estados.append(nodo.actualizar_estado(estado_combinado))
                else:
                    nuevos_estados.append(nodo.estado)
            
            estados_resultantes = nuevos_estados
            
        return estados_resultantes


class CapaAprendizajeCuantico:
    """Implementa la capa de aprendizaje cuántico con BayesLogic"""
    def __init__(self, dim_entrada, dim_salida):
        self.dim_entrada = dim_entrada
        self.dim_salida = dim_salida
        # Matriz de pesos cuánticos (incluye fase para representar aspectos cuánticos)
        self.pesos = np.random.rand(dim_salida, dim_entrada) + 1j * np.random.rand(dim_salida, dim_entrada)
        self.historial_exito = np.ones(dim_salida)
        self.historial_fallo = np.ones(dim_salida)
    
    def predecir(self, estado_entrada):
        """Realiza predicción usando estado cuántico de entrada"""
        # Producto de matrices para calcular salida
        salida_compleja = np.dot(self.pesos, estado_entrada)
        # Convertir a probabilidades (módulo al cuadrado)
        probabilidades = np.abs(salida_compleja)**2
        # Normalizar
        return probabilidades / np.sum(probabilidades)
    
    def actualizar_modelo(self, entrada, salida_deseada, tasa_aprendizaje=0.1):
        """Actualiza el modelo basado en retroalimentación"""
        prediccion = self.predecir(entrada)
        error = salida_deseada - prediccion
        
        # Actualiza matriz de pesos con corrección de error
        for i in range(self.dim_salida):
            for j in range(self.dim_entrada):
                # Actualiza tanto magnitud como fase
                self.pesos[i,j] += tasa_aprendizaje * error[i] * np.conj(entrada[j])
                
        # Actualiza historial bayesiano para acciones
        for i in range(self.dim_salida):
            if error[i] > 0:  # Si la predicción fue baja
                self.historial_exito[i] += abs(error[i])
            else:  # Si la predicción fue alta
                self.historial_fallo[i] += abs(error[i])
    
    def obtener_confianza_bayesiana(self):
        """Calcula la confianza bayesiana para cada salida"""
        total = self.historial_exito + self.historial_fallo
        confianza = self.historial_exito / total
        return confianza


class ArquitecturaIACuantica:
    """Clase principal que integra todos los componentes de la arquitectura"""
    def __init__(self, dim_entrada=4, dim_oculta=3, dim_salida=2):
        self.dim_entrada = dim_entrada
        self.dim_oculta = dim_oculta
        self.dim_salida = dim_salida
        
        # Inicializar componentes
        self.red_nodos = RedNodosFotonicos(num_nodos=dim_oculta, dimension=dim_entrada)
        self.capa_aprendizaje = CapaAprendizajeCuantico(dim_oculta, dim_salida)
    
    def procesar(self, datos_entrada):
        """Procesa datos a través de toda la arquitectura"""
        # Normalizar entradas (simulando codificación cuántica)
        entradas_norm = [x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x for x in datos_entrada]
        
        # Procesar a través de la red de nodos fotónicos
        estados_nodos = self.red_nodos.propagar_informacion(entradas_norm)
        
        # Extraer características de los estados resultantes
        caracteristicas = np.array([np.abs(estado)**2 for estado in estados_nodos])
        caracteristicas_aplanadas = caracteristicas.flatten()
        
        # Asegurar dimensionalidad correcta
        if len(caracteristicas_aplanadas) > self.dim_oculta:
            caracteristicas_vector = caracteristicas_aplanadas[:self.dim_oculta]
        else:
            # Extender si es necesario
            caracteristicas_vector = np.pad(caracteristicas_aplanadas, 
                                         (0, self.dim_oculta - len(caracteristicas_aplanadas)))
        
        # Normalizar para interpretación como estado cuántico
        if np.linalg.norm(caracteristicas_vector) > 0:
            caracteristicas_vector = caracteristicas_vector / np.linalg.norm(caracteristicas_vector)
        
        # Procesar a través de la capa de aprendizaje cuántico
        salida = self.capa_aprendizaje.predecir(caracteristicas_vector)
        
        return salida, estados_nodos
    
    def entrenar(self, datos_entrada, salidas_deseadas, iteraciones=100):
        """Entrena la arquitectura con datos de ejemplo"""
        historial_error = []
        
        for _ in range(iteraciones):
            error_total = 0
            
            for i, (entrada, salida_deseada) in enumerate(zip(datos_entrada, salidas_deseadas)):
                # Procesa a través de la red de nodos
                salida, estados = self.procesar([entrada])
                
                # Calcula error
                error = np.mean((salida - salida_deseada)**2)
                error_total += error
                
                # Actualiza modelo con retroalimentación
                self.capa_aprendizaje.actualizar_modelo(
                    np.array([np.abs(e)**2 for e in estados]).flatten()[:self.dim_oculta], 
                    salida_deseada
                )
            
            historial_error.append(error_total / len(datos_entrada))
            
        return historial_error
    
    def ejecutar_circuito_cuantico(self, datos_entrada):
        """Ejecuta un circuito cuántico basado en el estado actual del sistema"""
        # Procesar datos para obtener estados de nodos
        _, estados_nodos = self.procesar(datos_entrada)
        
        # Crear circuito para el primer nodo como demostración
        qc = self.red_nodos.nodos[0].crear_circuito_cuantico()
        
        # Simular y obtener resultado
        simulator = Aer.get_backend('statevector_simulator')
        job = execute(qc, simulator)
        resultado = job.result().get_statevector()
        
        return qc, resultado


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia de la arquitectura
    ia_cuantica = ArquitecturaIACuantica(dim_entrada=4, dim_oculta=3, dim_salida=2)
    
    # Datos de ejemplo (simples para demostración)
    X_train = [np.array([0.1, 0.2, 0.3, 0.4]), 
               np.array([0.4, 0.3, 0.2, 0.1]),
               np.array([0.5, 0.5, 0.1, 0.1])]
    
    y_train = [np.array([0.8, 0.2]), 
               np.array([0.3, 0.7]),
               np.array([0.5, 0.5])]
    
    # Entrenar el modelo
    historial_error = ia_cuantica.entrenar(X_train, y_train, iteraciones=50)
    
    # Visualizar error de entrenamiento
    plt.figure(figsize=(10, 6))
    plt.plot(historial_error)
    plt.title('Error de Entrenamiento')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error Cuadrático Medio')
    plt.grid(True)
    
    # Probar con un nuevo dato
    dato_prueba = np.array([0.2, 0.2, 0.3, 0.3])
    prediccion, _ = ia_cuantica.procesar([dato_prueba])
    print(f"Predicción para dato de prueba: {prediccion}")
    
    # Demostrar circuito cuántico
    circuito, estado_resultante = ia_cuantica.ejecutar_circuito_cuantico([dato_prueba])
    print(f"Estado cuántico resultante: {estado_resultante}")
    print("Circuito cuántico generado:")
    print(circuito)
    
    # Visualizar distribución de probabilidad del estado resultante
    probs = np.abs(estado_resultante)**2
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(probs)), probs)
    plt.title('Distribución de probabilidad del estado cuántico')
    plt.xlabel('Estado base')
    plt.ylabel('Probabilidad')
    plt.grid(True)
    
    # Mostrar confianza bayesiana aprendida
    confianza = ia_cuantica.capa_aprendizaje.obtener_confianza_bayesiana()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(confianza)), confianza)
    plt.title('Confianza Bayesiana por Dimensión de Salida')
    plt.xlabel('Dimensión')
    plt.ylabel('Confianza')
    plt.grid(True)
    
    # Mostrar todas las figuras
    plt.tight_layout()
    plt.show()
