from PIL import Image

def resize_image(input_image_path, output_image_path, scale=1/2):
    with Image.open(input_image_path) as image:
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        resized_image = image.resize((new_width, new_height))

        resized_image.save(output_image_path)
        
input_image = '/home/ubuntu/yifanlu/Chatsim2/ChatSim-release/data_utils/instruction_metashape/single_camera.jpg'
output_image = '/home/ubuntu/yifanlu/Chatsim2/ChatSim-release/data_utils/instruction_metashape/single_camera_resized.jpg'

resize_image(input_image, output_image)