import numpy as np
# from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


__all__ = ['COLORMAP', 'HIGHLIGHT_COLORMAP', 'COLORS', 'COLORS_COUNT',
          ]# 'generate_colors']


# Color creation routines
# -----------------------
def hue(H):
    H = H.reshape((-1, 1))
    R = np.abs(H * 6 - 3) - 1;
    G = 2 - np.abs(H * 6 - 2);
    B = 2 - np.abs(H * 6 - 4);
    return np.clip(np.hstack((R,G,B)), 0, 1)
    
def hsv_to_rgb(HSV):
    a = HSV[:,1].reshape((-1, 1))
    b = HSV[:,2].reshape((-1, 1))
    a = np.tile(a, (1, 3))
    b = np.tile(b, (1, 3))
    return ((hue(HSV[:,0]) - 1) * a + 1) * b

def generate_hsv(n0=20):
    H = np.linspace(0., 1., n0)
    i = np.arange(n0)
    H = H[~((i==5) | (i==7) | (i==10) | (i==12) | (i==15) |(i==17) | (i==18) | (i==19))]
    # H = H[((i==15) |(i==17) | (i==18) | (i==19))]

    H = np.repeat(H, 4)
    
    n = len(H)
    S = np.ones(n)
    V = np.ones(n)
    # change V for half of the colors
    V[1::2] = .75
    # change S for half of the colors
    S[2::4] = .75
    S[3::4] = .75
    
    hsv = np.zeros((n, 3))
    hsv[:,0] = H
    hsv[:,1] = S
    hsv[:,2] = V

    return hsv

    
# Global variables with all colors
# --------------------------------
# generate a list of RGB values for each color
hsv = generate_hsv()
hsv = np.clip(hsv, 0, 1)
COLORS = hsv_to_rgb(hsv)
COLORS = np.clip(COLORS, 0, 1)
COLORS_COUNT = len(COLORS)
step = 17  # needs to be prime with COLORS_COUNT
perm = np.mod(np.arange(0, step * 24, step), 24)
perm = np.hstack((2 * perm, 2 * perm + 1))
COLORMAP = COLORS[perm, ...]
COLORMAP = np.vstack(((1., 1., 1.), COLORMAP))

# Highlight color map
# decrease saturation, increase value
hsv[:,1] -= .5
hsv[:,2] += .5
hsv = np.clip(hsv, 0, 1)
hsv = hsv[perm, ...]
HIGHLIGHT_COLORMAP = hsv_to_rgb(hsv)
HIGHLIGHT_COLORMAP = np.vstack(((1., 1., 1.), HIGHLIGHT_COLORMAP))

def next_color(color):
    return np.mod(color, COLORS_COUNT) + 1

if __name__ == "__main__":
    def hsv_rect(hsv, coords):
        col = hsv_to_rgb(hsv)
        col = np.clip(col, 0, 1)
        rgb_rect(col, coords)
    
    def rgb_rect(rgb, coords):
        x0, y0, x1, y1 = coords
        a = 2./len(rgb)
        c = np.zeros((len(rgb), 4))
        c[:,0] = np.linspace(x0, x1-a, len(rgb))
        c[:,1] = y0
        c[:,2] = np.linspace(x0+a, x1, len(rgb))
        c[:,3] = y1
        rectangles(coordinates=c, color=rgb)
    
    from galry import *
    figure(constrain_navigation=False)
    rgb_rect(COLORMAP, (-1,0,1,1))
    rgb_rect(HIGHLIGHT_COLORMAP, (-1,-1,1,0))
    ylim(-1,1)
    show()
    
    
    
    # hsv = generate_hsv()
    # hsv_rect(hsv, (-1,0,1,1))
    
    # highlight
    # hsv[:,1] -= 0.5 # white -> color
    # hsv[:,2] += 0.5 # black -> white
    
    # hsv[:,1] -= 0.25 # white -> color
    # hsv[:,2] += 0.5 # black -> white
    
    # hsv_rect(hsv, (-1,-1,1,0))