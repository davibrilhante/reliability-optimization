import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString




width = 1350
height = 1350

bs_radius = 150

def hexagonalbsplacement(width:int, height:int,bs_radius:float, route=None, plot=False) -> list:
    hex_height = np.sqrt(bs_radius**2 - (bs_radius**2)/4)

    bs_rows = int(height/hex_height) - 1
    bs_columns = np.ceil(((width/bs_radius) - 0.5)/3)

    theta = np.pi/3

    bs_centers = []
    for x in range(int(bs_columns)):
        for y in range(bs_rows):
            center_x = (x*3 + 1)*bs_radius + bs_radius*1.5*(y%2)
            center_y = (y+1) * hex_height

            if center_x > width or center_y > height:
                continue

            points = []

            for i in range(6):
                xp = np.cos(theta*i)*bs_radius + center_x
                yp = np.sin(theta*i)*bs_radius + center_y
                points.append([xp, yp])

            polyg = Polygon(points)
            tolerance = bs_radius*np.sqrt(2)/2

            if type(route) == type(LineString([(0,0),(1,1)])):
                upperlimit = route.parallel_offset(tolerance,'left')
                lowerlimit = route.parallel_offset(tolerance,'right')

                if not (upperlimit.crosses(polyg) or lowerlimit.crosses(polyg)):
                    continue

            bs_centers.append([center_x, center_y])

            if plot:
                plt.scatter(center_x, center_y)
                plt.text(center_x-50, center_y-50, str(len(bs_centers)-1))

                i, j = polyg.exterior.xy
                plt.plot(i, j)

    if plot:
        plt.xticks(np.linspace(0,1200,1200/bs_radius+1))
        plt.yticks(np.linspace(0,1200,1200/bs_radius+1))
        plt.xlim(0,1200)
        plt.ylim(0,1200)
        plt.grid()
        plt.show()

    return bs_centers

if __name__ == "__main__":
    route = LineString([(0,0),(1200,1200)])
    coord = hexagonalbsplacement(width, height, bs_radius, route, True)
    print(coord)
