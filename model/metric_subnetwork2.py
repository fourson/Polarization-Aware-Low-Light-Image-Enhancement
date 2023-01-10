from .metric_utils.psnr import PSNR
from .metric_utils.ssim import SSIM


psnr = PSNR().cuda()
ssim = SSIM(channel=3).cuda()

ssim_gray = SSIM(channel=1).cuda()


