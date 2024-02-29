import cv2

def merge_videos(video_path1, video_path2, output_path):
    # 打开输入视频
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # 获取视频的帧率和帧大小
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, min(fps1, fps2), (width1 + width2, max(height1, height2)))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果两个视频都已结束，跳出循环
        if not ret1 and not ret2:
            break
        
        # 如果其中一个视频结束，用黑色帧代替
        if not ret1:
            frame1 = 255 * np.ones((height1, width1, 3), dtype=np.uint8)
        if not ret2:
            frame2 = 255 * np.ones((height2, width2, 3), dtype=np.uint8)
        
        # 调整帧的高度，使其相等
        if height1 != height2:
            frame1 = cv2.resize(frame1, (width1, height2))
            frame2 = cv2.resize(frame2, (width2, height2))

        # 水平拼接两帧
        merged_frame = cv2.hconcat([frame1, frame2])
        
        # 写入新视频
        out.write(merged_frame)

    cap1.release()
    cap2.release()
    out.release()

video_path1 = "/dssg/home/acct-agrtkx/agrtkx/wz/Inpaint-Anything/demo_videos/1346_ori.mp4"
video_path2 = "/dssg/home/acct-agrtkx/agrtkx/wz/Inpaint-Anything/demo_videos/1346_inpaint.mp4"
output_path = "/dssg/home/acct-agrtkx/agrtkx/wz/Inpaint-Anything/demo_videos/merged_video_1346.mp4"
merge_videos(video_path1, video_path2, output_path)
