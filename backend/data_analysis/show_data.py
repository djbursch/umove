import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

client = MongoClient()
db = client['keypoints']
keypoints = db.keypoints

all_points = keypoints.find({"user_id":"jack1"})
pts = np.asarray(all_points[0]['keypoints'])

y_axis = []
for i in range(len(pts)):
    y_axis.append(np.mean(pts[i, :, 0, :, 0]))

x_axis = []
for i in range(len(pts)):
    x_axis.append(i)

x_axis = np.asarray(x_axis)
xnew = np.linspace(x_axis.min(), x_axis.max(), int(len(x_axis)/3)) 
spl = make_interp_spline(x_axis, y_axis, k=3) 
power_smooth = spl(xnew)

fig, ax = plt.subplots()
ax.plot(power_smooth)
ax.invert_yaxis()
plt.ylabel('Round Off Back Handspring to Layout')
plt.show()