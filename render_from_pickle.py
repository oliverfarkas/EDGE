import bpy
import os
import pickle
import sys
import math

# Add the directory containing this script to sys.path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

import helper

def move_and_rotate_light():
    light = bpy.data.objects.get("Light")
    
    if light is not None:
        light.location.x = 4.07625
        light.location.y = 1.00545
        light.location.z = -5.90386
        
        light.rotation_euler[0] = math.radians(37.261)
        light.rotation_euler[1] = math.radians(3.16371)
        light.rotation_euler[2] = math.radians(106.936)
    else:
        print("No light found in the scene!")

def move_and_rotate_camera():
    camera = bpy.data.objects.get("Camera")
    
    if camera is not None:
        camera.location.x = 0
        camera.location.y = 0
        camera.location.z = -18
        
        camera.rotation_euler[0] = math.radians(180)
        camera.rotation_euler[1] = 0
        camera.rotation_euler[2] = 0
    else:
        print("No camera found in the scene!")

def save_blender_file(file_path):
    bpy.ops.wm.save_as_mainfile(filepath=file_path)
    
def load_smpl_model(pkl_path):
    extension = pkl_path.split(".")[-1]
    if extension != "pkl":
        print(f"File should be a pickle file with .pkl extension, got {extension}")
        return

    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)
        smpl_params = {"smpl_poses":data["smpl_poses"],
                       "smpl_trans":data["smpl_trans"]}
                    #    "smpl_trans":data["smpl_trans"] / data["smpl_scaling"][0]}

    print("Read smpl from {}".format(pkl_path))

    gender = "male"
    path = os.path.dirname(os.path.realpath(__file__))
    objects_path = os.path.join(path, "load_smpl_addon/data", "smpl-model-20200803.blend", "Object")
    object_name = "SMPL-mesh-" + gender

    bpy.ops.wm.append(filename=object_name, directory=str(objects_path))

    object_name = bpy.context.selected_objects[-1].name
    obj = bpy.data.objects[object_name]
    obj.select_set(True)

    filename = os.path.basename(pkl_path)
    bpy.ops.object.text_add(location=(0, 0, obj.dimensions.z + 0.2))
    text_obj = bpy.context.object
    text_obj.data.body = filename
    text_obj.parent = obj
    text_obj.data.size = 1
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'

    text_obj.rotation_euler[0] = 0
    text_obj.rotation_euler[1] = math.pi
    text_obj.rotation_euler[2] = math.pi

    move_and_rotate_camera()
    move_and_rotate_light()

    rotation_euler_xyz, translation_front_up_right = helper.GetAnimation(smpl_params)
    for b in obj.pose.bones:
        b.rotation_mode = "XYZ"
    obj.animation_data_create()
    obj.animation_data.action = bpy.data.actions.new(name="SMPL motion")

    for bone_name, bone_data in rotation_euler_xyz.items():
        fcurve_0 = obj.animation_data.action.fcurves.new(
            data_path=f'pose.bones["{bone_name}"].rotation_euler', index=0
        )
        fcurve_1 = obj.animation_data.action.fcurves.new(
            data_path=f'pose.bones["{bone_name}"].rotation_euler', index=1
        )
        fcurve_2 = obj.animation_data.action.fcurves.new(
            data_path=f'pose.bones["{bone_name}"].rotation_euler', index=2
        )
        for frame in range(1, bone_data.shape[0]+1):
            k0 = fcurve_0.keyframe_points.insert(frame=frame, value=bone_data[frame-1, 0])
            k1 = fcurve_1.keyframe_points.insert(frame=frame, value=bone_data[frame-1, 1])
            k2 = fcurve_2.keyframe_points.insert(frame=frame, value=bone_data[frame-1, 2])

    fcurve_x = obj.animation_data.action.fcurves.new(
        data_path=f'pose.bones["Pelvis"].location', index=0
    )
    fcurve_y = obj.animation_data.action.fcurves.new(
        data_path=f'pose.bones["Pelvis"].location', index=1
    )
    fcurve_z = obj.animation_data.action.fcurves.new(
        data_path=f'pose.bones["Pelvis"].location', index=2
    )
    for frame in range(1, translation_front_up_right.shape[0]+1):
        k0 = fcurve_y.keyframe_points.insert(frame=frame, value=translation_front_up_right[frame-1, 0])
        k1 = fcurve_z.keyframe_points.insert(frame=frame, value=translation_front_up_right[frame-1, 1])
        k2 = fcurve_x.keyframe_points.insert(frame=frame, value=translation_front_up_right[frame-1, 2])

    obj.animation_data.action.frame_start = 1
    obj.animation_data.action.frame_end = bone_data.shape[0]

    bpy.context.scene.render.fps = 30

    return {'FINISHED'}

def remove_default_cube():
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

def setup_quicktime_h264_render(output_path):
    # Set the render file output path
    bpy.context.scene.render.filepath = output_path

    # Set render resolution (optional, adjust as needed)
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100

    # Set the frame range for the animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 100

    # Set the file format to FFmpeg video
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

    # Set the container to QuickTime
    bpy.context.scene.render.ffmpeg.format = 'QUICKTIME'

    # Set the video codec to H.264
    bpy.context.scene.render.ffmpeg.codec = 'H264'

    # Set the output quality (adjust as needed)
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'HIGH'

    # Set the encoding speed (optional)
    bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

    # Set audio codec (optional, if your scene has audio)
    bpy.context.scene.render.ffmpeg.audio_codec = 'AAC'
    bpy.context.scene.render.ffmpeg.audio_bitrate = 192  # kbps

def render_animation():
    # Render the animation
    bpy.ops.render.render(animation=True)

def process_all_pkl_files(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    remove_default_cube()
    dir_name = os.path.basename(os.path.normpath(input_dir))
    output_blend = os.path.join(output_dir, f"{dir_name}.blend")
    output_mov = os.path.join(output_dir, f"{dir_name}.mov")
    # Loop through all .pkl files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            pkl_path = os.path.join(input_dir, filename)
            print(f"Processing {pkl_path}")
            load_smpl_model(pkl_path)
    
    save_blender_file(output_blend)
    # setup_quicktime_h264_render(output_mov)
    # render_animation()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: blender --background --python render_from_pickle.py -- <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[-2]
    output_dir = sys.argv[-1]
    
    process_all_pkl_files(input_dir, output_dir)
