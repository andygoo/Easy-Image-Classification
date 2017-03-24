import os
import subprocess
import argparse

def video2images(in_dir, out_dir, crop_size, out_size, framerate, video_exts):



    video_exts = tuple(ext.lower() for ext in video_exts)


    video_count = 0

   
    for current_dir, dir_names, file_names in os.walk(in_dir):

        relative_path = os.path.relpath(current_dir, in_dir)


        new_dir = os.path.join(out_dir, relative_path)


        if not os.path.exists(new_dir):
            os.makedirs(new_dir)


        for file_name in file_names:
            if file_name.lower().endswith(video_exts):
                in_file = os.path.join(current_dir, file_name)
                file_root, file_ext = os.path.splitext(file_name)

                new_file_name = file_root + "-%4d.jpg"


                new_file_path = os.path.join(new_dir, new_file_name)


                new_file_path = os.path.normpath(new_file_path)

   

             
                cmd = "avconv -i {0} -r {1} -vf crop={2}:{2} -vf scale={3}:{3} -qscale 2 {4}"


                cmd = cmd.format(in_file, framerate, crop_size, out_size, new_file_path)


                subprocess.call(cmd, shell=True)
                video_count += 1


              

    print("Number of videos converted: {0}".format(video_count))



if __name__ == "__main__":
   
 
  
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--indir", required=True,
                        help="input directory where videos are located")

    parser.add_argument("--outdir", required=True,
                        help="output directory where images will be saved")

    parser.add_argument("--crop", required=True, type=int,
                        help="the input videos are first cropped to CROP:CROP pixels")

    parser.add_argument("--size", required=True, type=int,
                        help="the input videos are then resized to SIZE:SIZE pixels")

    parser.add_argument("--rate", required=False, type=int, default=5,
                        help="the number of frames to convert per second")

    parser.add_argument("--exts", required=False, nargs="+",
                        help="list of extensions for video-files e.g. .mts .mp4")

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Get the arguments.
    in_dir = args.indir
    out_dir = args.outdir
    crop_size = args.crop
    out_size = args.size
    framerate = args.rate
    video_exts = args.exts

    if video_exts is None:
        # Default extensions for video-files.
        video_exts = (".MTS", ".mp4", ".MOV")
    else:

        video_exts = tuple(video_exts)

 
    print("Convert videos to images.")
    print("- Input dir: " + in_dir)
    print("- Output dir: " + out_dir)
    print("- Crop width and height: {0}".format(crop_size))
    print("- Resize width and height: {0}".format(out_size))
    print("- Frame-rate: {0}".format(framerate))
    print("- Video extensions: {0}".format(video_exts))
    print()

   
    video2images(in_dir=in_dir, out_dir=out_dir,
                 crop_size=crop_size, out_size=out_size,
                 framerate=framerate, video_exts=video_exts)


