import torch
from torch import nn


class SiLU(nn.Module):
    """
    SiLU (Sigmoid Linear Unit, также известная как Swish) — современная функция активации для нейросетей и LLM.

    Назначение:
    -----------
    - Формирует плавную нелинейную активацию: SiLU(x) = x * sigmoid(x).
    - Активно используется во всех новых архитектурах для больших языковых моделей (PaLM, LLaMA, Mistral, GPT-4 и др.).
    - Дает лучший поток градиентов по сравнению с ReLU, SELU, GELU в глубоких слоях — позволяет делать сети больше и глубже.

    Мотивация и свойства:
    ---------------------
    - SiLU объединяет свойства identity (для больших x) и ReLU (для отрицательных x, где есть затухание), но более плавно.
    - Позволяет проходить отрицательным значениям, а не "обрубает" как ReLU.
    - Better for optimization and training dynamics in deep LLMs, приводит к более богатым аппроксимациям.

    Математическая формула:
    -----------------------
        SiLU(x) = x * sigmoid(x)
    где sigmoid(x) = 1 / (1 + exp(-x))

    Сравнение с другими активациями:
    --------------------------------
    - ReLU(x): max(0, x) — простая отсечка
    - GELU(x): плавная вероятностная активация (используется в BERT/GPT-2)
    - SiLU(x): плавная альтернатива, часто лучше в современных LLM
    - Swish (Ramachandran et al., 2017) = SiLU

    Args:
    -----
    Нет learnable параметров, чисто функциональная активация.

    Пример использования:
    ---------------------
        >>> silu = SiLU()
        >>> x = torch.tensor([-2.0, 0.0, 2.0])
        >>> print(silu(x))  # тензор с элементами [-0.2384, 0.0, 1.7616] (примерно)

    References:
    -----------
    - Ramachandran et al., "Searching for Activation Functions", 2017: https://arxiv.org/abs/1710.05941
    - LLaMA: https://arxiv.org/abs/2302.13971
    - Swish в TensorFlow: https://arxiv.org/abs/1710.05941
    - Сравнение всех актив. функций: https://paperswithcode.com/method/silu
    """

    def forward(self, x: torch.Tensor):
        """
        Применяет SiLU активацию ко всем компонентам тензора (x * sigmoid(x)).
    
        Args:
        -----
        x : torch.Tensor
            Входной тензор любой формы.
    
        Returns:
        --------
        torch.Tensor — тензор той же формы, каждый элемент преобразован по формуле SiLU(x).
    
        Пример:
        -------
            >>> silu = SiLU()
            >>> x = torch.linspace(-3, 3, 7)
            >>> y = silu(x)
        """
        return torch.sigmoid(x) * x
