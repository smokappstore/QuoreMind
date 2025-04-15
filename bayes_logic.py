"""
Autor: Jacobo Tlacaelel Mina Rodríguez (optimizado y documentación por Claude, Gemini y ChatGPT)
Fecha: 16/03/2025
Versión: cuadrante-coremind v1.2.1
"""

import logging
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import numpy as np

# Configuración del log para imprimir información por consola.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BayesLogic:
    """
    Clase para calcular probabilidades y seleccionar acciones basadas en el teorema de Bayes.
    
    Esta clase proporciona funcionalidad para evaluar decisiones basadas en estados del sistema
    caracterizados por su entropía, coherencia e influencia probabilística, utilizando
    inferencia bayesiana para seleccionar la acción óptima.
    
    Atributos:
        EPSILON (float): Valor mínimo para evitar divisiones por cero.
        HIGH_ENTROPY_THRESHOLD (float): Umbral para considerar alta entropía.
        HIGH_COHERENCE_THRESHOLD (float): Umbral para considerar alta coherencia.
        ACTION_THRESHOLD (float): Umbral para decidir acción (1 si es mayor, 0 si es menor).
        
    Ejemplo de uso:
        >>> logic = BayesLogic()
        >>> result = logic.calculate_probabilities_and_select_action(
        ...     entropy=0.9, coherence=0.7, prn_influence=0.5, action=1
        ... )
        >>> print(f"Acción seleccionada: {result['action_to_take']}")
    """
    def __init__(self, 
                 epsilon: float = 1e-6, 
                 high_entropy_threshold: float = 0.8, 
                 high_coherence_threshold: float = 0.6, 
                 action_threshold: float = 0.5) -> None:
        """
        Inicializa la clase BayesLogic con parámetros personalizables.
        
        Args:
            epsilon: Valor mínimo para evitar divisiones por cero.
            high_entropy_threshold: Umbral para considerar alta entropía.
            high_coherence_threshold: Umbral para considerar alta coherencia.
            action_threshold: Umbral para decidir acción.
        """
        self._validate_thresholds(epsilon, high_entropy_threshold, high_coherence_threshold, action_threshold)
        
        self.EPSILON = epsilon
        self.HIGH_ENTROPY_THRESHOLD = high_entropy_threshold
        self.HIGH_COHERENCE_THRESHOLD = high_coherence_threshold
        self.ACTION_THRESHOLD = action_threshold
        
        logger.info(f"BayesLogic inicializado con umbrales: entropía={high_entropy_threshold}, "
                   f"coherencia={high_coherence_threshold}, acción={action_threshold}")

    def _validate_thresholds(self, epsilon: float, high_entropy_threshold: float, 
                            high_coherence_threshold: float, action_threshold: float) -> None:
        """
        Valida que los umbrales estén dentro de rangos aceptables.
        
        Args:
            epsilon: Valor mínimo para evitar divisiones por cero.
            high_entropy_threshold: Umbral para alta entropía.
            high_coherence_threshold: Umbral para alta coherencia.
            action_threshold: Umbral para decidir acción.
            
        Raises:
            ValueError: Si algún umbral está fuera del rango válido.
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon debe ser positivo, valor recibido: {epsilon}")
        
        for threshold_name, threshold_value in [
            ("high_entropy_threshold", high_entropy_threshold),
            ("high_coherence_threshold", high_coherence_threshold),
            ("action_threshold", action_threshold)
        ]:
            if not 0 <= threshold_value <= 1:
                raise ValueError(f"{threshold_name} debe estar entre 0 y 1, valor recibido: {threshold_value}")

    def calculate_posterior_probability(self, prior_a: float, prior_b: float, 
                                       conditional_b_given_a: float) -> float:
        """
        Calcula la probabilidad posterior utilizando el teorema de Bayes.
        
        Formula: P(A|B) = P(B|A) * P(A) / P(B)
        
        Args:
            prior_a: Probabilidad previa de A, P(A).
            prior_b: Probabilidad previa de B, P(B).
            conditional_b_given_a: Probabilidad condicional de B dado A, P(B|A).
            
        Returns:
            La probabilidad posterior P(A|B).
        """
        self._validate_probability(prior_a, "prior_a")
        self._validate_probability(prior_b, "prior_b")
        self._validate_probability(conditional_b_given_a, "conditional_b_given_a")
        
        # Evita división por cero
        prior_b = max(prior_b, self.EPSILON)
        return (conditional_b_given_a * prior_a) / prior_b

    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        """
        Calcula la probabilidad condicional P(A|B) = P(A,B) / P(B).
        
        Args:
            joint_probability: Probabilidad conjunta P(A,B).
            prior: Probabilidad previa P(B).
            
        Returns:
            La probabilidad condicional P(A|B).
        """
        self._validate_probability(joint_probability, "joint_probability")
        self._validate_probability(prior, "prior")
        
        # Evita división por cero
        prior = max(prior, self.EPSILON)
        return joint_probability / prior

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        """
        Determina la probabilidad previa basada en el nivel de entropía.
        
        Args:
            entropy: Valor de entropía entre 0 y 1.
            
        Returns:
            Probabilidad previa basada en entropía.
        """
        self._validate_probability(entropy, "entropy")
        return 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.1

    def calculate_high_coherence_prior(self, coherence: float) -> float:
        """
        Determina la probabilidad previa basada en el nivel de coherencia.
        
        Args:
            coherence: Valor de coherencia entre 0 y 1.
            
        Returns:
            Probabilidad previa basada en coherencia.
        """
        self._validate_probability(coherence, "coherence")
        return 0.6 if coherence > self.HIGH_COHERENCE_THRESHOLD else 0.2

    def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
        """
        Calcula la probabilidad conjunta en función de coherencia, acción e influencia PRN.
        
        Args:
            coherence: Nivel de coherencia entre 0 y 1.
            action: Acción a evaluar (0 o 1).
            prn_influence: Influencia probabilística entre 0 y 1.
            
        Returns:
            Probabilidad conjunta calculada.
        """
        self._validate_probability(coherence, "coherence")
        self._validate_probability(prn_influence, "prn_influence")
        self._validate_action(action)
        
        if coherence > self.HIGH_COHERENCE_THRESHOLD:
            if action == 1:
                return prn_influence * 0.8 + (1 - prn_influence) * 0.2
            else:
                return prn_influence * 0.1 + (1 - prn_influence) * 0.7
        return 0.3

    def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, 
                                                 prn_influence: float, action: int) -> Dict[str, float]:
        """
        Integra los diferentes cálculos probabilísticos para seleccionar una acción.
        
        Este método realiza una serie de cálculos bayesianos para determinar la acción
        más apropiada según los valores de entropía, coherencia e influencia PRN.
        
        Args:
            entropy: Nivel de entropía del sistema (0-1).
            coherence: Nivel de coherencia del sistema (0-1).
            prn_influence: Influencia probabilística (0-1).
            action: Acción a evaluar (0 o 1).
            
        Returns:
            Diccionario con las probabilidades calculadas y la acción seleccionada.
        """
        self._validate_probability(entropy, "entropy")
        self._validate_probability(coherence, "coherence")
        self._validate_probability(prn_influence, "prn_influence")
        self._validate_action(action)
        
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)
        
        # Cálculo de probabilidad condicional P(B|A)
        conditional_b_given_a = (prn_influence * 0.7 + (1 - prn_influence) * 0.3
                                if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.2)
        
        # Cálculo de probabilidad posterior P(A|B)
        posterior_a_given_b = self.calculate_posterior_probability(
            high_entropy_prior, high_coherence_prior, conditional_b_given_a
        )
        
        # Cálculo de probabilidad conjunta P(A,B)
        joint_probability_ab = self.calculate_joint_probability(coherence, action, prn_influence)
        
        # Cálculo de probabilidad condicional P(acción|B)
        conditional_action_given_b = self.calculate_conditional_probability(
            joint_probability_ab, high_coherence_prior
        )
        
        # Determinación de la acción a tomar
        action_to_take = 1 if conditional_action_given_b > self.ACTION_THRESHOLD else 0
        
        logger.info(f"Acción seleccionada: {action_to_take} con probabilidad: {conditional_action_given_b:.4f}")
        
        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }
    
    def _validate_probability(self, value: float, parameter_name: str) -> None:
        """
        Valida que un valor esté dentro del rango válido para una probabilidad [0,1].
        
        Args:
            value: Valor a validar.
            parameter_name: Nombre del parámetro para mensajes de error.
            
        Raises:
            ValueError: Si el valor está fuera del rango [0,1].
        """
        if not 0 <= value <= 1:
            raise ValueError(f"El parámetro {parameter_name} debe estar entre 0 y 1, valor recibido: {value}")
    
    def _validate_action(self, action: int) -> None:
        """
        Valida que una acción sea 0 o 1.
        
        Args:
            action: Valor de acción a validar.
            
        Raises:
            ValueError: Si la acción no es 0 o 1.
        """
        if action not in [0, 1]:
            raise ValueError(f"La acción debe ser 0 o 1, valor recibido: {action}")
    
    def visualize_probabilities(self, result_dict: Dict[str, float], title: str = "Probabilidades Bayesianas") -> None:
        """
        Visualiza las probabilidades calculadas como un gráfico de barras.
        
        Args:
            result_dict: Diccionario con las probabilidades calculadas.
            title: Título para el gráfico.
        """
        # Filtrar solo valores numéricos de probabilidad (excluyendo action_to_take)
        prob_dict = {k: v for k, v in result_dict.items() if isinstance(v, float)}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(prob_dict.keys(), prob_dict.values(), color='skyblue')
        
        # Añadir etiquetas de valores
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Probabilidad')
        ax.set_title(title)
        ax.axhline(y=self.ACTION_THRESHOLD, color='r', linestyle='-', label=f'Umbral de Acción ({self.ACTION_THRESHOLD})')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def sensitivity_analysis(self, parameter_to_vary: str, 
                            range_start: float = 0.0, 
                            range_end: float = 1.0,
                            steps: int = 20,
                            fixed_params: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza un análisis de sensibilidad variando un parámetro y observando cómo cambia la decisión.
        
        Args:
            parameter_to_vary: Nombre del parámetro a variar ('entropy', 'coherence', o 'prn_influence').
            range_start: Valor inicial del parámetro.
            range_end: Valor final del parámetro.
            steps: Número de pasos para la variación.
            fixed_params: Diccionario con valores fijos para los otros parámetros.
            
        Returns:
            Tupla con dos arrays: valores del parámetro variado y probabilidades condicionales resultantes.
            
        Raises:
            ValueError: Si el parámetro a variar no es válido.
        """
        if parameter_to_vary not in ['entropy', 'coherence', 'prn_influence']:
            raise ValueError(f"Parámetro a variar no válido: {parameter_to_vary}. "
                           f"Debe ser 'entropy', 'coherence', o 'prn_influence'")
        
        # Valores por defecto si no se proporcionan
        if fixed_params is None:
            fixed_params = {'entropy': 0.5, 'coherence': 0.5, 'prn_influence': 0.5, 'action': 1}
        
        # Asegurar que todos los parámetros necesarios estén presentes
        required_params = {'entropy', 'coherence', 'prn_influence', 'action'}
        for param in required_params:
            if param != parameter_to_vary and param not in fixed_params:
                fixed_params[param] = 0.5  # Valor por defecto
                
        # Crear array de valores para el parámetro variado
        param_values = np.linspace(range_start, range_end, steps)
        probability_values = np.zeros(steps)
        
        # Calcular probabilidades para cada valor
        for i, value in enumerate(param_values):
            params = fixed_params.copy()
            params[parameter_to_vary] = value
            
            result = self.calculate_probabilities_and_select_action(
                entropy=params['entropy'],
                coherence=params['coherence'],
                prn_influence=params['prn_influence'],
                action=params['action']
            )
            
            probability_values[i] = result['conditional_action_given_b']
        
        return param_values, probability_values
    
    def plot_sensitivity_analysis(self, parameter_to_vary: str, 
                                 range_start: float = 0.0, 
                                 range_end: float = 1.0,
                                 steps: int = 20,
                                 fixed_params: Optional[Dict[str, float]] = None) -> None:
        """
        Realiza y visualiza un análisis de sensibilidad.
        
        Args:
            parameter_to_vary: Nombre del parámetro a variar.
            range_start: Valor inicial del parámetro.
            range_end: Valor final del parámetro.
            steps: Número de pasos para la variación.
            fixed_params: Diccionario con valores fijos para los otros parámetros.
        """
        param_values, probability_values = self.sensitivity_analysis(
            parameter_to_vary, range_start, range_end, steps, fixed_params
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, probability_values, marker='o', linestyle='-')
        plt.axhline(y=self.ACTION_THRESHOLD, color='r', linestyle='--', 
                   label=f'Umbral de Acción ({self.ACTION_THRESHOLD})')
        
        plt.xlabel(f'Valor de {parameter_to_vary}')
        plt.ylabel('Probabilidad Condicional')
        plt.title(f'Análisis de Sensibilidad: Efecto de {parameter_to_vary} en la Decisión')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Sombrear región de acción = 1
        above_threshold = probability_values >= self.ACTION_THRESHOLD
        plt.fill_between(param_values, self.ACTION_THRESHOLD, probability_values, 
                        where=above_threshold, alpha=0.3, color='green', 
                        label='Región Acción = 1')
        
        # Sombrear región de acción = 0
        below_threshold = probability_values < self.ACTION_THRESHOLD
        plt.fill_between(param_values, probability_values, self.ACTION_THRESHOLD, 
                        where=below_threshold, alpha=0.3, color='red',
                        label='Región Acción = 0')
        
        plt.legend()
        plt.tight_layout()
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia de BayesLogic con los umbrales por defecto
    bayes_logic = BayesLogic()
    
    # Calcular probabilidades y seleccionar acción
    result = bayes_logic.calculate_probabilities_and_select_action(
        entropy=0.85,
        coherence=0.75,
        prn_influence=0.6,
        action=1
    )
    
    print("Resultados del cálculo:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Visualizar las probabilidades
    bayes_logic.visualize_probabilities(result, "Análisis de Probabilidades Bayesianas")
    
    # Realizar análisis de sensibilidad
    print("\nRealizando análisis de sensibilidad para el parámetro 'coherence'...")
    fixed_params = {'entropy': 0.85, 'prn_influence': 0.6, 'action': 1}
    bayes_logic.plot_sensitivity_analysis('coherence', fixed_params=fixed_params)
