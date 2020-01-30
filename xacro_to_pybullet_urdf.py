import os
import re
import shutil
import sys

if len(sys.argv) < 2:
    print("Error: Specify filename!")
    print("Example: python xacro_to_pybullet_urdf.py '/home/julien/MASTER_THESIS_ROS/ROS_wrapper/catkin_ws/src/moma/moma_description/urdf/mopa.urdf.xacro'")
    exit()
# NOTE: Environment in which xacro file is located must be sourced!
xacro_file = sys.argv[1]


# Obtain robot name
robot_name = xacro_file.split('/')[-1].split('.')[0]

# Generate URDF
urdf = os.popen("xacro {} --inorder".format(xacro_file)).read()

# Prepare regex, find all files from packages
regex = re.compile("package://(.*?)/([\w|/|\.|-]*)")
files = regex.findall(urdf)

# Prepare folder
shutil.rmtree('urdf/robot/{}'.format(robot_name), ignore_errors=True)
os.makedirs('urdf/robot/{}'.format(robot_name))
os.makedirs('urdf/robot/{}/meshes'.format(robot_name))

# Copy file to new folder and replace in urdf
for match in files:
    package_name, relpath = match
    original_string = "package://{}/{}".format(package_name, relpath)
    newstring = "meshes/{}/{}".format(package_name, relpath)
    urdf = urdf.replace(original_string, newstring)

    package_path = os.popen("rospack find {}".format(package_name)).read()
    package_path = package_path.replace('\n', '')
    abs_path = os.path.join(package_path, relpath)
    relfolder = '/'.join(newstring.split('/')[:-1])
    newpath = 'urdf/robot/{}/{}'.format(robot_name, relfolder)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    shutil.copy(abs_path, 'urdf/robot/{}/{}'.format(robot_name, newstring))

# Save to file
filename = 'urdf/robot/{}/{}.urdf'.format(robot_name, robot_name)
with open(filename, "w") as text_file:
    text_file.write(urdf)

print("Done. Check folder {}!".format('urdf/robot/{}'.format(robot_name)))
