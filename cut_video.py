from moviepy.video.io.VideoFileClip import VideoFileClip
CHECKER = True 

if CHECKER == False:
    clip = VideoFileClip("../nymeria_dataset/20230607_s0_james_johnson_act0_e72nhq/Nymeria_v0.0_20230607_s0_james_johnson_act0_e72nhq_preview_rgb.mp4")

    # Information about the video
    # Note: The codec 'libx264' is commonly used for MP4 files.
    print(f"Duration: {clip.duration} seconds")
    print(f"FPS: {clip.fps}")
    print(f"Size: {clip.size} pixels")
    print(f"Resolution: {clip.w}x{clip.h}")
    print(f"Audio present: {clip.audio is not None}")
    print(f"File name: {clip.filename}")

    short_clip = clip.subclipped(30, 60)
    output_file = "output_short_30-60.mp4"
    short_clip.write_videofile(output_file, codec="libx264")

    # Close the clip to release resources
    clip.close()

else:
    # Information about the short clip
    s_clip = VideoFileClip("test_data/Nymeria/output_short_30-60.mp4")

    print(f"Short clip duration: {s_clip.duration} seconds")
    print(f"Short clip FPS: {s_clip.fps}")
    print(f"Short clip size: {s_clip.size} pixels")
    print(f"Short clip resolution: {s_clip.w}x{s_clip.h}")
    print(f"Short clip audio present: {s_clip.audio is not None}")
    print(f"Short clip file name: {s_clip.filename}")

    s_clip.close()
