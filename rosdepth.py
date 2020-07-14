import rospy
import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')

args = parser.parse_args()


bridge = CvBridge()
class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("/usb_cam/image_raw2",Image,queue_size=10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

  def callback(self,data):
    with tf.Graph().as_default():
      try:
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)

      height = args.height
      width = args.width
      placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}
      
      with tf.variable_scope("model") as scope:
        model = pydnet(placeholders)

      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())

      loader = tf.train.Saver()
      saver = tf.train.Saver()
      

      #cv2.imshow("Image window", img)
      #cv2.waitKey(1)

      #print(img)  
      with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
          
        img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
        img = np.expand_dims(img, 0)
        start = time.time()
        disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
        end = time.time()
        disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
        disp_color2 = cv2.cvtColor(disp_color, cv2.COLOR_BGR2GRAY)      
        disp_color2 = cv2.bitwise_not(disp_color2)
        print("Time: " + str(end - start))
        del img
        del disp
      
      
      try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(disp_color2, "32FC1"))
      except CvBridgeError as e:
        print(e)
      #cv2.imshow("Image window", cv_image)
      #      cv2.waitKey(3)




def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)