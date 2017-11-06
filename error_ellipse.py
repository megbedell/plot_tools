'''
A matplotlib-based function to overplot an elliptical error contour from the covariance matrix.
Copyright 2017 Megan Bedell (Flatiron).

Citations: Joe Kington (https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py),
           Vincent Spruyt (http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/)
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def error_ellipse(ax, xc, yc, cov, sigma=1, **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_artist(ellipse)
    
if __name__ == '__main__':
    #-- Example usage -----------------------
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
            mean=(1,1), cov=[[5., 4.],[4., 6.]], size=100
            )
    x, y = points.T
    cov = np.cov(x,y, rowvar=False)
    
    # Plot the raw points...
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color='k')
    # Plot three error ellipses
    error_ellipse(ax, np.mean(x), np.mean(y), cov, ec='red')
    error_ellipse(ax, np.mean(x), np.mean(y), cov, sigma=2, ec='green')
    error_ellipse(ax, np.mean(x), np.mean(y), cov, sigma=3, ec='blue')
    plt.show()