from main import num_px, parameters, predict, plt, np, classes
from PIL import Image
import imageio

# 用自己的猫图进行测试
my_image = "my_image.jpg"  # change this to the name of your image file
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(imageio.imread(fname))
my_image = np.array(Image.fromarray(image).resize((num_px, num_px))).reshape((num_px * num_px * 3, 1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")