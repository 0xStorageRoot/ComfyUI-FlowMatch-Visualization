import numpy as np
import inspect
import torch
import matplotlib.pyplot as plt
import io
from typing import Optional, Tuple, Dict, Union, List
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from .scheduler import FlowUniPCMultistepScheduler, FlowDPMSolverMultistepScheduler, get_sampling_sigmas
class CalculatorSteps:

    @staticmethod
    def calculate_steps(scheduler_style: str,
                        steps: int,
                        shift: float,
                        boundary: float,
                        skip: int,
                        cfg_skip: float,
                        ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        根据给定参数计算每一个steps对应的时间戳，并按boundary分组
        
        Args:
            scheduler_style: 调度器样式
            steps: 总采样步数 (N)
            shift: shift 参数
            boundary: 边界参数
            skip: 跳过的步数
            cfg_skip: CFG跳过参数
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: 
            {
                'above_boundary': (sigmas, t_values) - 大于steps*boundary的值,
                'below_boundary': (sigmas, t_values) - 其余的值
            }
        """
        scheduler_calculate = get_wan_scheduler(scheduler_style, shift)
        #目前只支持了FlowMatchEulerDiscreteScheduler
        if isinstance(scheduler_calculate, FlowMatchEulerDiscreteScheduler):
            timesteps, _ = retrieve_timesteps(scheduler_calculate, steps, mu=1)
        elif isinstance(scheduler_calculate, FlowUniPCMultistepScheduler):
            timesteps = scheduler_calculate.set_timesteps(steps, shift=shift) 
        elif isinstance(scheduler_calculate, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(
                scheduler_calculate,
                sigmas=sampling_sigmas)
        else:
            timesteps, _ = retrieve_timesteps(scheduler_calculate, steps, timesteps)
        
        # (i,t,expert,skip_start,skip_cfg)
        step_info_array = []
        for i, t in enumerate(timesteps):
            # t >= boundary * scheduler.config.num_train_timesteps 原代码中判定切换transformer的逻辑
            # 检查scheduler是否有config属性，如果有则使用config.num_train_timesteps，否则直接使用num_train_timesteps
            num_timesteps = 1000
            expert = "high" if t >= boundary * num_timesteps else "low"
            
            #不开启teacache不会有skip_start效果
            # pipeline.transformer.enable_teacache(coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload)
            # i <= skip时skip_start = true否则false
            skip_start = i <= skip
            
            # bs = len(x)
            # if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio): 原代码中判定跳过cfg只处理空白提示词的逻辑,后面还会一样进行noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            # i >= steps*(1-cfg_skip)时skip_cfg = true否则false
            skip_cfg = i >= steps * (1 - cfg_skip)
            
            step_info_array.append((i, t, expert, skip_start, skip_cfg))

        return step_info_array

    @staticmethod
    def plot_step_info(step_info_array, title="Steps Visualization", save_path=None):
        """
        使用step_info_array绘制散点图
        
        Args:
            step_info_array: 包含(i, t, expert, skip_start, skip_cfg)信息的列表
            title: 图表标题
            save_path: 保存图片的路径，如果为None则不保存
        """
        import matplotlib
        
        # 设置中文字体，避免字体缺失警告
        try:
            # 尝试使用常见的中文字体
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except:
            # 如果设置字体失败，使用英文标签
            pass

        
        # 分离数据 - 每个点只分类一次
        high_expert_solid = []  # high expert 实心点
        high_expert_hollow = []  # high expert 空心点 (cfg_skip)
        low_expert_solid = []   # low expert 实心点
        low_expert_hollow = []  # low expert 空心点 (cfg_skip)
        skip_start_points = []  # skip_start 灰色空心点
        
        for i, t, expert, skip_start, skip_cfg in step_info_array:
            # skip_start为true的点单独绘制为灰色空心点
            if skip_start:
                skip_start_points.append((i, t))
                continue
                
            # 每个点只归类到一个类别中
            if expert == "high":
                if skip_cfg:
                    high_expert_hollow.append((i, t))
                else:
                    high_expert_solid.append((i, t))
            else:  # expert == "low"
                if skip_cfg:
                    low_expert_hollow.append((i, t))
                else:
                    low_expert_solid.append((i, t))
        
        # 创建图表
        plt.figure(figsize=(10, 8))
        
        # 绘制skip_start灰色空心点
        if skip_start_points:
            skip_x, skip_y = zip(*skip_start_points)
            plt.scatter(skip_x, skip_y, c='none', marker='o', s=60, 
                       edgecolors='gray', linewidth=2, 
                       label=f'Skip Start ({len(skip_start_points)})')
        
        # 绘制实心点
        if high_expert_solid:
            high_x, high_y = zip(*high_expert_solid)
            plt.scatter(high_x, high_y, c='blue', marker='o', s=60, 
                       label=f'High Expert ({len(high_expert_solid)})')
        
        if low_expert_solid:
            low_x, low_y = zip(*low_expert_solid)
            plt.scatter(low_x, low_y, c='red', marker='o', s=60, 
                       label=f'Low Expert ({len(low_expert_solid)})')
        
        # 绘制空心点 (cfg_skip) - 使用更粗的边框和更大的尺寸来突出显示
        if high_expert_hollow:
            high_cfg_x, high_cfg_y = zip(*high_expert_hollow)
            plt.scatter(high_cfg_x, high_cfg_y, c='none', marker='o', s=80, 
                       edgecolors='blue', linewidth=3, 
                       label=f'High Expert/CFG Skip ({len(high_expert_hollow)})')
        
        if low_expert_hollow:
            low_cfg_x, low_cfg_y = zip(*low_expert_hollow)
            plt.scatter(low_cfg_x, low_cfg_y, c='none', marker='o', s=80, 
                       edgecolors='red', linewidth=3, 
                       label=f'Low Expert/CFG Skip ({len(low_expert_hollow)})')
        
        # 设置图表属性
        plt.xlabel('steps')
        plt.ylabel('timesteps')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        

        pil_image = Image.open(buf)
        pil_image = pil_image.convert('RGB')
        

        image_array = np.array(pil_image)
        print(f"Image array shape: {image_array.shape}")
        
        # 确保图像是RGB格式 (height, width, 3)
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_array.shape}")
        
        # 转换为张量格式: (batch, height, width, channels) - ComfyUI标准格式
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度: (1, height, width, 3)
        
        print(f"Final tensor shape: {image_tensor.shape}")
        
        plt.close()
        buf.close()

        return (image_tensor,)


# copied from videox-fun/comfyui/wan2_1/nodes.py
def get_wan_scheduler(sampler_name, shift):
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    scheduler_kwargs = {
        "num_train_timesteps": 1000,
        "shift": 5.0,
        "use_dynamic_shifting": False,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
    }
    scheduler_kwargs['shift'] = shift
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, scheduler_kwargs)
    )
    return scheduler

# copied from videox-fun/utils/utils.py
def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps