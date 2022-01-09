#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

cam_center = pipeline.createColorCamera()
cam_center.setPreviewSize(200, 200)
cam_center.setBoardSocket(dai.CameraBoardSocket.RGB) # center camera on the oak-d
cam_center.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_center.setInterleaved(False)
cam_center.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setConfidenceThreshold(200)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setRectifyEdgeFillColor(0)
cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)

# Create outputs
xout_center = pipeline.createXLinkOut()
xout_center.setStreamName('center')
cam_center.video.link(xout_center.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName('depth')
stereo.disparity.link(xout_depth.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    # starts automatically
    #device.startPipeline()

    # Output queues will be used to get the grayscale frames from the outputs defined above
    q_center = device.getOutputQueue(name="center", maxSize=1, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    frame_center = None
    frame_depth_orig = None
    frame_depth = None

    counter = 0

    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_center = q_center.tryGet()
        in_depth = q_depth.tryGet()

        if in_center is not None:
            frame_center = in_center.getCvFrame()
            frame_center = cv2.resize(frame_center, (640, 480), interpolation=cv2.INTER_AREA)
        
        if in_depth is not None:
            frame_depth_orig = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
            frame_depth = np.ascontiguousarray(frame_depth_orig)
            frame_depth = (frame_depth * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
            frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

        # show the frames if available
            cv2.imshow("center", frame_center)
        if frame_depth is not None:
            cv2.imshow("depth", frame_depth)

        if frame_depth is not None and frame_center is not None and in_depth is not None:
            if in_depth.getHeight() == 480:
                counter = counter + 1
                if counter >= 10:
                    break

        if cv2.waitKey(1) == ord('q'):
            break

print("got a frame")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = []
ys = []
zs = []
cs = []

for y in tqdm(range(in_depth.getHeight())):
    for x in range(80, 480): #in_depth.getWidth()):
        depth = frame_depth_orig[y, x]
        color = frame_center[y, x]
        if (depth > 50):
            xs.append(x)
            ys.append(y)
            zs.append(depth)
            cs.append((color[2]/255, color[1]/255, color[0]/255))

PLOTEVERY = 3
ax.scatter(xs[::PLOTEVERY], ys[::PLOTEVERY], zs[::PLOTEVERY], marker='.', color=cs[::PLOTEVERY])
plt.show()