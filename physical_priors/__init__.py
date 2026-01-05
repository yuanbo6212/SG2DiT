"""
Physical Priors for LFDT-Fusion
频域和各向异性物理先验模块
"""

from .wavelets import DWTForward, DWTInverse, FrequencyEncoder
from .anisotropic import StructureTensorAnalyzer, AnisotropicGuidanceModule
from .encoder import PhysicalPriorEncoder
from .losses import PhysicalPriorLoss
from .utils import visualize_priors, compute_metrics

"""
Physical Priors for LFDT-Fusion
频域和各向异性物理先验模块
"""

# 核心模块（必需的）
try:
    from .wavelets import DWTForward, DWTInverse, FrequencyEncoder

    _wavelets_available = True
except ImportError as e:
    print(f"Warning: Wavelets module import failed: {e}")
    _wavelets_available = False

try:
    from .anisotropic import StructureTensorAnalyzer, AnisotropicGuidanceModule

    _anisotropic_available = True
except ImportError as e:
    print(f"Warning: Anisotropic module import failed: {e}")
    _anisotropic_available = False

# 编码器模块（可选）
try:
    from .encoder import PhysicalPriorEncoder

    _encoder_available = True
except ImportError as e:
    print(f"Warning: Encoder module not available: {e}")
    _encoder_available = False

# 损失函数模块（可选）
try:
    from .losses import PhysicalPriorLoss

    _losses_available = True
except ImportError as e:
    print(f"Warning: Losses module not available: {e}")
    _losses_available = False

# 工具函数模块（可选）
try:
    from .utils import visualize_priors, compute_metrics

    _utils_available = True
except ImportError as e:
    print(f"Warning: Utils module not available: {e}")


    # 提供占位函数
    def visualize_priors(*args, **kwargs):
        print("Warning: visualize_priors function not available")
        return None


    def compute_metrics(*args, **kwargs):
        print("Warning: compute_metrics function not available")
        return {}


    _utils_available = False

# 动态构建 __all__ 列表
__all__ = []

if _wavelets_available:
    __all__.extend(['DWTForward', 'DWTInverse', 'FrequencyEncoder'])

if _anisotropic_available:
    __all__.extend(['StructureTensorAnalyzer', 'AnisotropicGuidanceModule'])

if _encoder_available:
    __all__.append('PhysicalPriorEncoder')

if _losses_available:
    __all__.append('PhysicalPriorLoss')

# 工具函数总是可用（即使是占位函数）
__all__.extend(['visualize_priors', 'compute_metrics'])

# 版本信息
__version__ = '1.0.0'


# 模块可用性检查函数
def check_module_availability():
    """检查各模块的可用性"""
    status = {
        'wavelets': _wavelets_available,
        'anisotropic': _anisotropic_available,
        'encoder': _encoder_available,
        'losses': _losses_available,
        'utils': _utils_available
    }

    print("Physical Priors Module Status:")
    for module, available in status.items():
        status_str = "✅ Available" if available else "❌ Not Available"
        print(f"  {module}: {status_str}")

    return status


# 获取可用功能列表
def get_available_features():
    """获取当前可用的功能列表"""
    features = []

    if _wavelets_available:
        features.append("Frequency Domain Analysis (Wavelets)")

    if _anisotropic_available:
        features.append("Anisotropic Structure Analysis")

    if _encoder_available:
        features.append("Physical Prior Encoding")

    if _losses_available:
        features.append("Physical Prior Loss Functions")

    if _utils_available:
        features.append("Visualization and Metrics")

    return features


# 初始化时的状态检查（可选，仅在debug模式下）
import os

if os.environ.get('PHYSICAL_PRIORS_DEBUG', '').lower() == 'true':
    print(f"Physical Priors v{__version__} initialized")
    check_module_availability()