# %ESTTHETA Estimate projection line of best fit
# %
# % Options::
# % 'mi'               Estimate similarity using mutual information
# % 'zncc'             Estimate similarity using normalized cross correlation
# % 'zssd'             Estimate similarity using SSD
# % 'mask',M           Use only pixels corresponding to mask image M
# % 'geometricmean'    Compute chromaticity using geom mean of R,G,B
# % 'approxlog',A      Use approximation to log with alpha value A
# % 'offset',OFF       Add offset OFF to images to eliminate Inf, Nan
# % 'sharpen',M        Sharpening transform
# %
# % See also INVARIANT, SIMILARITY

from machinevisiontoolbox import Image
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import base, Polygon2

def esttheta(im, sharpen=None):

    k_region = pickregion(im)

    imcol = im.column()

    z = imcol[k_region, :]
    print(z.shape)
    # k = find(in);
    plt.show(block=True)

#     if isempty(opt.mask)
#         fprintf('pick shaded/lit region of same material\nclick several points on the perimeter of the region then hit RETURN\n'); beep;
#         k_region = pickregion(im);
#         drawnow
#         im = im2col(im, k_region);
#     else
#         im = im2col(im, opt.mask);
#     end

#     if ~isempty(opt.sharpen)
#         im = double(im) * opt.sharpen;
#         im = max(0, im);
#     end
    
#     theta_v = [0:0.02:pi];
#     sim = [];

#     %for theta = theta_v
#     for i=1:numel(theta_v)
#         theta = theta_v(i);
#         gs = invariant(im, theta, args{:});
#         k = isinf(gs) | isnan(gs);
#         sim = [sim std(gs(~k))];
#     end

#     figure
#     plot(theta_v, sim);
#     xlabel('invariant line angle (rad)');
#     ylabel('invariance image variance');

#     [~,k] = min(sim);
#     fprintf('best angle = %f rad\n', theta_v(k));
#     %idisp( invariant(gs, theta) );
    
#     if nargout > 0
#         th = theta_v(k);
#     end
# end

def pickregion(im):

    im.disp()

    clicks = plt.ginput(n=-1)

    xy = np.array(clicks)
    print(xy)
    
    base.plot_poly(xy.T, 'g', close=True)

    polygon = Polygon2(xy.T)
    polygon.plot('g')
    
    X, Y = im.meshgrid()
    inside = polygon.contains(np.c_[X.ravel(), Y.ravel()].T)
    
    return inside

if __name__ == "__main__":

    im = Image.Read('parks.png')

    esttheta(im)