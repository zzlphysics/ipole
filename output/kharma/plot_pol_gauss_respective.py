"""

  Makes images from hdf5 output of ipole.
  2019.07.10 gnw

$ python ipole_plot.py path/to/images/*h5

$ ffmpeg -framerate 8 -i dump%*.png -s:v 1280x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p out.mp4
"""

import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter


## configuration / plot parameters

FOV_UNITS = "muas"  # can be set to "muas" or "M" (case sensitive)

## EVPA_CONV not yet implemented! this will only work for observer convention!
EVPA_CONV = "EofN"  # can be set fo "EofN" or "NofW" 



## no need to touch anything below this line
def colorbar(mappable):
    """ 使用科学计数法格式化colorbar，只显示5个刻度 """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # 获取数据范围
    vmin = mappable.get_clim()[0]
    vmax = mappable.get_clim()[1]
    
    # 生成5个均匀分布的刻度值
    ticks = np.linspace(vmin, vmax, 5)
    
    # 创建colorbar并设置刻度格式
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels(['{:.1e}'.format(tick) for tick in ticks])
    
    return cbar

# def colorbar(mappable):
#   """ the way matplotlib colorbar should have been implemented """
#   from mpl_toolkits.axes_grid1 import make_axes_locatable
#   ax = mappable.axes
#   fig = ax.figure
#   divider = make_axes_locatable(ax)
#   cax = divider.append_axes("right", size="5%", pad=0.05)
#   return fig.colorbar(mappable, cax=cax)

# 当模块作为主程序运行时
if __name__ == "__main__":
  
  # 遍历命令行参数中除脚本名外的每个文件名
  for fname in sys.argv[1:]:

    # 如果文件名不以.h5结尾，则跳过当前文件
    if fname[-3:] != ".h5": 
      continue

    # 输出当前正在处理的文件名
    print("plotting {0:s}".format(fname))

    # 加载HDF5文件
    hfp = h5py.File(fname,'r')    
    # 从文件中读取所需元数据
    dx = hfp['header']['camera']['dx'][()]
    dsource = hfp['header']['dsource'][()]
    lunit = hfp['header']['units']['L_unit'][()]
    # 计算视场大小（以微弧秒为单位）
    fov_muas = dx / dsource * lunit * 2.06265e11
    # 读取缩放因子
    scale = hfp['header']['scale'][()]
    freq = hfp['header']['freqcgs'][()]
    # 初始化电矢量位置角
    evpa_0 = 'W'
    # 如果文件中存在电矢量位置角，则读取之
    if 'evpa_0' in hfp['header']:
      evpa_0 = hfp['header']['evpa_0'][()]
    # 读取未偏振光强度图像
    unpol = np.copy(hfp['unpol']).transpose((1,0))
    # 读取偏振光强度图像
    imagep = np.copy(hfp['pol']).transpose((1,0,2))
    # 从 hfp 对象中通过键 'pol' 获取数据。这里假设 hfp 是一个已经打开的 HDF5 文件或类似支持字典访问方式的对象。
    # 使用 np.copy() 函数创建获取到的数据的一个副本，并将这个副本赋值给变量 imagep。
    # 对 imagep 进行转置操作，改变其维度顺序。原数据的维度顺序假设为 (宽度, 高度, 通道数)，转置后变为 (高度, 宽度, 通道数)。

    # 分离偏振图像的各分量
    I = np.flip(imagep[:,:,0], axis=0)
    Q = np.flip(imagep[:,:,1], axis=0)
    U = np.flip(imagep[:,:,2], axis=0)
    V = np.flip(imagep[:,:,3], axis=0)
    # 关闭HDF5文件
    hfp.close()

    # FWHM = 20 muas
    # FWHM = 2*sqrt(2*ln(2)) * sigma 
    # 20 muas = 2*sqrt(2*ln(2))*sigma = 2*sqrt(2*ln(2)) * npix * (dw/D) * rad2muas 
    # sigma(pix) = npix = ...... 
    nx = I.shape[0]
    rad2muas = np.pi/180/3600/1000000
    sigma = 20 / (2 * np.sqrt(2 * np.log(2))) / ((dx*lunit/nx)/(dsource)/rad2muas)

    c_cgs = 2.99792458e10
    k_cgs = 1.38064852e-16
    Tb = c_cgs**2/(2*freq**2*k_cgs) * I

    convolved_Tb = gaussian_filter(Tb, sigma)
    convolved_Q = gaussian_filter(Q, sigma)
    convolved_U = gaussian_filter(U, sigma)

    # 如果在这里对图像进行rotation，和在进行计算时修改rotcam参数得到的偏振图像是否一致？

    Tb_rot = rotate(Tb, 288, reshape=False)
    Tb_rot[Tb_rot<0] = 0
    con_rot = rotate(convolved_Tb, 288, reshape=False)
    con_rot[con_rot<0] = 0

    # 设置图像显示范围
    if FOV_UNITS == "muas":
      extent = [ -fov_muas/2, fov_muas/2, -fov_muas/2, fov_muas/2 ]
    elif FOV_UNITS == "M":
      extent = [ -dx/2, dx/2, -dx/2, dx/2 ]
    else:
      # 如果视场单位未被识别，则输出错误信息并退出
      print("! unrecognized units for FOV {0:s}. quitting.".format(FOV_UNITS))

    # command line output
    I_flux = I.sum()*scale
    unpol_flux = unpol.sum()*scale
    Q_flux = Q.sum()*scale 
    U_flux = U.sum()*scale
    V_flux = V.sum()*scale
    LP = 100.*np.sqrt(Q_flux**2+U_flux**2)/I_flux
    CP = 100.*V_flux/I_flux
    print("Flux [Jy]:    {0:g} {1:g}".format(I_flux, unpol_flux))
    print("I,Q,U,V [Jy]: {0:g} {1:g} {2:g} {3:g}".format(I_flux,Q_flux,U_flux,V_flux))
    print("LP [%]:       {0:g}".format(LP))
    print("CP [%]:       {0:g}".format(CP))
    evpatot = 180./3.14159*0.5*np.arctan2(U.sum(),Q.sum())
    if evpa_0 == "W":
      evpatot += 90. 
      if evpatot > 90.:
        evpatot -= 180
    if EVPA_CONV == "NofW":
      evpatot += 90.
      if evpatot > 90.:
        evpatot -= 180
    print("EVPA [deg]:   {0:g}".format(evpatot))


    plt.close('all')

    fig1, ax1 = plt.subplots(figsize=(6,5), dpi=300)
    Imax = I.max() #/np.sqrt(1.5)
    im1 = ax1.imshow(I, cmap='afmhot', vmin=0., vmax=Imax, origin='upper', extent=extent)
    colorbar(im1)
    ax1.set_title("Stokes I [cgs]     Flux = {0:.1f} Jy".format(I_flux))
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$\mu as$')
    ax1.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_I.png"))

    fig2, ax2 = plt.subplots(figsize=(6,5), dpi=300)
    Qmax = np.abs(Q.max()) if np.abs(Q.max()) > np.abs(Q.min()) else np.abs(Q.min())
    Qmin = -Qmax
    im2 = ax2.imshow(Q, cmap='seismic', vmin=Qmin, vmax=Qmax, origin='upper', extent=extent)
    colorbar(im2)
    ax2.set_title("Stokes Q [cgs]     Flux = {0:.1f} Jy".format(Q_flux))
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$\mu as$')
    ax2.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Q.png"))

    fig3, ax3 = plt.subplots(figsize=(6,5), dpi=300)
    Umax = np.abs(U.max()) if np.abs(U.max()) > np.abs(U.min()) else np.abs(U.min())
    Umin = -Umax
    im3 = ax3.imshow(U, cmap='seismic', vmin=Umin, vmax=Umax, origin='upper', extent=extent)
    colorbar(im3)
    ax3.set_title("Stokes U [cgs]     Flux = {0:.1f} Jy".format(U_flux))
    ax3.set_aspect('equal')
    ax3.set_xlabel(r'$\mu as$')
    ax3.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_U.png"))
    fig4, ax4 = plt.subplots(figsize=(6,5), dpi=300)
    Vmax = np.abs(V.max()) if np.abs(V.max()) > np.abs(V.min()) else np.abs(V.min())
    Vmin = -Vmax
    im4 = ax4.imshow(V, cmap='seismic', vmin=Vmin, vmax=Vmax, origin='upper', extent=extent)
    colorbar(im4)
    ax4.set_title("Stokes V [cgs]     CP = {0:.1f}%".format(CP))
    ax4.set_aspect('equal')
    ax4.set_xlabel(r'$\mu as$')
    ax4.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_V.png"))

    fig5, ax5 = plt.subplots(figsize=(6,5), dpi=300)
    Tbmax = Tb.max()
    im5 = ax5.imshow(Tb, cmap='afmhot', vmin=0., vmax=Tbmax, origin='upper', extent=extent)
    colorbar(im5)
    ax5.set_title(r"$T_b$ [K]     Flux = {0:.1f} Jy".format(I_flux))
    ax5.set_aspect('equal')
    ax5.set_xlabel(r'$\mu as$')
    ax5.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Tb.png"))

    fig6, ax6 = plt.subplots(figsize=(6,5), dpi=300)
    Icmax = convolved_Tb.max()
    im6 = ax6.imshow(convolved_Tb, cmap='afmhot', vmin=0., vmax=Icmax, origin='upper', extent=extent)
    colorbar(im6)
    ax6.set_title(r"$T_b$(Gaussian blur) [K]     Flux = {0:.1f} Jy".format(I_flux))
    ax6.set_aspect('equal')
    ax6.set_xlabel(r'$\mu as$')
    ax6.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Tb_gauss.png"))

    fig7, ax7 = plt.subplots(figsize=(6,5), dpi=300)
    im7 = ax7.imshow(Tb_rot, cmap='afmhot', vmin=0., vmax=Tbmax, origin='upper', extent=extent)
    colorbar(im7)
    ax7.set_title("Tb(PA288°) [K]     Flux = {0:.1f} Jy".format(I_flux))
    ax7.set_aspect('equal')
    ax7.set_xlabel(r'$\mu as$')
    ax7.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Tb_rot.png"))

    fig8, ax8 = plt.subplots(figsize=(6,5), dpi=300)
    im8 = ax8.imshow(con_rot, cmap='afmhot', vmin=0., vmax=Icmax, origin='upper', extent=extent)
    colorbar(im8)
    ax8.set_title(r"$T_b$(Gaussian blur, PA288°) [K]     Flux = {0:.1f} Jy".format(I_flux))
    ax8.set_aspect('equal')
    ax8.set_xlabel(r'$\mu as$')
    ax8.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Tb_gauss_rot.png"))

    fig9, ax9 = plt.subplots(figsize=(6,5), dpi=300)
    im9 = ax9.imshow(Tb, cmap='afmhot', vmin=0., vmax=Tbmax, origin='upper', extent=extent)
    colorbar(im9)
    evpa = (180./3.14159)*0.5*np.arctan2(U,Q)
    if evpa_0 == "W":
      evpa += 90.
      evpa[evpa > 90.] -= 180.
    if EVPA_CONV == "NofW":
      evpa += 90.
      evpa[evpa > 90.] -= 180.
    # quiver on intensity
    npix = Tb.shape[0]
    xs = np.linspace(-fov_muas/2,fov_muas/2,npix)
    ys = np.linspace(fov_muas/2,-fov_muas/2,npix)
    Xs,Ys = np.meshgrid(xs,ys)
    lpscal = np.max(np.sqrt(Q*Q+U*U))
    vxp = np.sqrt(Q*Q+U*U)*np.sin(evpa*np.pi/180.)/lpscal
    vyp = -np.sqrt(Q*Q+U*U)*np.cos(evpa*np.pi/180.)/lpscal
    skip = int(npix/32) 
    ax9.quiver(Xs[::skip,::skip],Ys[::skip,::skip],vxp[::skip,::skip],vyp[::skip,::skip], 
      headwidth=1, headlength=1, 
      width=0.005,
      color='#00ff00', 
      units='width', 
      scale=4,
      pivot='mid')
    
    ax9.set_title(r"$T_b$ [k]    LP = {0:.1f}%".format(LP))
    ax9.set_aspect('equal')
    ax9.set_xlabel(r'$\mu as$')
    ax9.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Tb_quiver.png"))

    fig10, ax10 = plt.subplots(figsize=(6,5), dpi=300)
    im10 = ax10.imshow(convolved_Tb, cmap='afmhot', vmin=0., vmax=Icmax, origin='upper', extent=extent)
    colorbar(im10)
    evpa_con = (180./3.14159)*0.5*np.arctan2(convolved_U,convolved_Q)
    if evpa_0 == "W":
      evpa_con += 90.
      evpa_con[evpa_con > 90.] -= 180.
    if EVPA_CONV == "NofW":
      evpa_con += 90.
      evpa_con[evpa_con > 90.] -= 180.
    # quiver on intensity
    npix = convolved_Tb.shape[0]
    xs = np.linspace(-fov_muas/2,fov_muas/2,npix)
    ys = np.linspace(fov_muas/2,-fov_muas/2,npix)
    Xs,Ys = np.meshgrid(xs,ys)
    lpscal_con = np.max(np.sqrt(convolved_Q*convolved_Q+convolved_U*convolved_U))
    vxp = np.sqrt(convolved_Q*convolved_Q+convolved_U*convolved_U)*np.sin(evpa_con*np.pi/180.)/lpscal_con
    vyp = -np.sqrt(convolved_Q*convolved_Q+convolved_U*convolved_U)*np.cos(evpa_con*np.pi/180.)/lpscal_con
    skip = int(npix/16) 
    ax10.quiver(Xs[::skip,::skip],Ys[::skip,::skip],vxp[::skip,::skip],vyp[::skip,::skip], 
      headwidth=1, headlength=1, 
      width=0.005,
      color='#00ff00', 
      units='width', 
      scale=4,
      pivot='mid')
    
    ax10.set_title(r"$T_b$(Gaussian blur)     LP = {0:.1f}%".format(LP))
    ax10.set_aspect('equal')
    ax10.set_xlabel(r'$\mu as$')
    ax10.set_ylabel(r'$\mu as$')
    plt.savefig(fname.replace(".h5","_Tb_gauss_quiver.png"))

