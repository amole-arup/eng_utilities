"""Plotting Utilities

These are for providing visualisations for testing.

In order not to consider matplotlib as a requirement,
the import of matplotlib and the functions are held
within a try-except, and dummy functions are provided.
"""

from eng_utilities.polyline_utilities import bounding_box2D


# Plotting Utility
try:
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D # for 3D plots
    from matplotlib import __version__ as mpl_ver
    from matplotlib import patches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    print(f'MatPlotLib version {mpl_ver} is installed')
    
    try:
        from mpl_toolkits.mplot3d import Axes3D # for 3D plots
    except ImportError:
        print('mplot3d is imported, but out of date')
        from mpl_toolkits.mplot3d import axes3D as Axes3D # for 3D plots

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import PolyCollection
    from matplotlib.axes import Axes


    #def beamPlot(outline, polygeom, rebar_geom, title = "Beam Plot"):
    def multi_plot2D(polyline_list = [], point_lists = [], polygon_colls = None, 
        title = "Multi Plot (2D)", **kwargs):
        """Generates a plot of a list of polylines, each defined 
        by a list of (x, y) tuplets, and a list of lists of points (x, y)
        of rebar definitions - defined by tuplets of 
        area and location (area, x, y).
        If a value is missing then simply provide an empty list, e.g.
        multi_plot(sec1, [], rebar_geom1) 
        The things that can be set using kwargs are defined in the documentation for subplot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html
        
        Args:
            polyline_list (list): a collection (list) of lists of vertices 
                to be plotted as lines
            point_lists (list): a collection (list) of lists of vertices 
                to be plotted as points
            polygon_colls (list): a collection (list) of lists of vertices, 
                to be plotted as solid polygons
            title (str): plot title
        
        Returns:
            Axes: 
        """

        figsize = kwargs.get('figsize',(6.4, 4.8))
        fig, ax = plt.subplots(1,1,figsize=figsize) # note we must use plt.subplots, not plt.subplot
        ax.set_aspect('equal', adjustable='box')
        ax.set_frame_on(False)
        plt.title(title)
        for polyline in polyline_list:
            a = [] ; b = []
            for pt in polyline:
                (x,y,*_) = pt
                a.append(x)
                b.append(y)
            plt.plot(a, b)
        
        for point_list in point_lists:
            if len(point_list) > 0:
                pts = [(x, y) for x, y, *_  in point_list]
                c, d = zip(*pts)  
                plt.plot(c, d, 'o')
        
        # for polyline_coll in polyline_colls:
        # you need to pass a collection here, not just one polygon, 
        # you can solve that by putting brackets around a list of vertices
        # poly = PolyCollection([verts], facecolor='r', edgecolor='g', closed=True)
        if polygon_colls is not None:
            #print('plotting polygon')
            p_gons = [[(x, y) for x, y, *_ in polygon] for polygon in polygon_colls]
            (xmin, ymin),(xmax, ymax) = bounding_box2D(sum(p_gons, []))            
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            poly = PolyCollection(p_gons, facecolor='r', edgecolor='g', alpha=0.2, closed=True)
            ax.add_artist(poly)
        
        return fig


    def multi_plot3D(polyline_list = [], point_lists = None, polyline_colls = None, style='-o', *args, **kwargs):
        """This function uses matplotlib to plot lists of 3D tuples (x, y, z)
        style is a string to describe the line style, e.g. 
            '-' for lines only, 
            'g-' for green lines only, 
            'o-' for dots and lines, 
            'o' for dots without lines
        In addition, point lists and other collections may be plotted using `point_lists` and `polyline_colls`
        """
        fig = plt.figure()
        #ax = plt.axes()
        ax = fig.add_subplot(111, projection='3d')
        for polyline in polyline_list:
            x_range, y_range, z_range = zip(*polyline)
            ax.plot(x_range, y_range, z_range, style)
        
        if point_lists is not None:
            ax.add_collection3d(Poly3DCollection([[[x, y, z] for x, y, z in tri] for tri in point_lists], alpha = 0.5))
            
        if polyline_colls is not None:
            ax.add_collection3d(polyline_colls)
            
        return plt


except:
    # Dummy definitions
    def multi_plot2D(*args, **kwargs):
        print('MatPlotLib is not installed')


    def multi_plot3D(*args, **kwargs):
        print('MatPlotLib is not installed')


try:
    # %pip install triangle
    # tr.triangulate
    # A = dict(vertices=np.array(((0, 0), (1, 0), (1, 1), (0, 1))))
    # B = tr.triangulate(A, 'qa0.01') # or 'qa0.1'
    # tr.compare(plt, A, B)
    # plt.show()
    # A = dict(vertices=vertices, segments=segments, regions=regions)
    # B = tr.triangulate(A, 'pA')
    import triangle as tr
    print('Triangle installed.')
except:
    print('Triangle is not installed')
    

