# 指针表盘识别使用说明

这份脚本用来读半圆形指针表盘，输出 0 到 50 的整数值。  
它适合做批量截图识别，也可以在调试模式下把中间过程画出来，方便判断识别是否可靠。

## 1. 运行环境

- Python 3.9 及以上
- 依赖库：opencv-python、numpy

安装依赖：

    pip install opencv-python numpy

## 2. 最常用的三种用法

单张识别：

    python gauge_reader.py 指针.png

单张识别并保存结果到文件：

    python gauge_reader.py --out result.txt 指针.png

单张识别并导出调试图：

    python gauge_reader.py --debug 指针.png

调试图默认保存为 debug_result.png，图上会标出圆心、候选线段、最终指针和读数。

## 3. 多张图片输入

可以一次传多张图，程序会对每张图先识别，再取众数作为稳定值：

    python gauge_reader.py 图1.png 图2.png 图3.png

如果你加了 --debug，会额外导出每一帧的调试图，文件名类似：
- debug_result_1.png
- debug_result_2.png
- debug_result_3.png

## 4. 命令行参数说明

- --debug：输出可视化调试图，方便排查识别问题
- --out 文件路径：把最终结果写入文本文件
- 图片路径：支持一个或多个

参数顺序可以灵活一些，只要图片路径最终都带上即可。

## 5. 输出结果说明

- 正常识别：控制台输出 0 到 50 的整数
- 识别失败：输出 None
- 图片读不到：会打印警告并跳过该图片

## 6. 识别效果和图片质量建议

这部分是实战里最影响成功率的点：

- 尽量让表盘主体占画面大部分，别太远
- 避免大角度倾斜和强反光
- 保证指针和刻度区域有清晰边缘
- 同一设备建议固定拍摄角度和距离

如果现场环境变化比较大，建议每次识别时连拍 3 到 5 张，再用多帧众数结果，通常会稳很多。

## 7. 常见问题

Q：为什么明明有图却输出 None？  
A：常见原因是边缘不清晰、反光太重、指针太短或底边干扰太强。先用 --debug 看看候选线段和最终指针是否画对。

Q：Windows 下中文路径会不会读图失败？  
A：这个脚本已经做了兼容处理，中文路径可以直接用。

Q：结果偶尔跳动怎么办？  
A：优先使用多图输入，让程序取众数；其次尽量固定拍摄条件。

## 8. 文件说明

- gauge_reader.py：识别主脚本
- result.txt：示例输出文件
- debug_result.png：单图调试模式下的可视化结果

## 9. 接入 ROS

这个脚本已经留了两个对外接口，方便你直接接到 ROS 节点里：

- `process_gauge_image(image, output_callback=None)`：输入一张 `numpy.ndarray` 格式的 BGR 图像，返回读数
- `process_gauge_image_debug(image, output_callback=None)`：调试模式接口，额外返回可视化图
- `publish_gauge_result(value, output_callback=None, debug=None)`：把读数和调试信息交给外部发布逻辑

回调函数统一接收两个参数：

- `value`：识别结果，类型是整数或 `None`
- `debug`：调试信息字典，可以按需取里面的 `work_image`、`crop_rect`、`center`、`debug_vis` 等字段

一个很小的接法示意如下：

    def on_gauge_result(value, debug):
        print("value =", value)
        # 这里可以把 value 发布到 ROS topic
        # 如果 debug 里有 debug_vis，也可以顺手发布调试图

    value = process_gauge_image(cv_image, on_gauge_result)

如果你是用 `sensor_msgs/Image` 订阅相机话题，先把 ROS 图像转成 OpenCV 的 BGR 图，再调用上面的接口就行。识别结果建议单独发一个数字话题，调试图则单独发一个图像话题，这样后面排查会比较清楚。
