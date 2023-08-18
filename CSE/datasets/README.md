# CounterfactualMNIST
Following [Kenny and Keane, 2021], we sampled a part of
the MNIST dataset by selecting classes of images where the
digits can be masked such that they can be converted to a
different digit with high confidence. For example, the digit
9 can be converted to a 7 by masking certain regions such
that it appears as a 7. A counterfactual explanation based
on analysis of subobject regions is expected to identify the
exact regions that need to be masked in a digit to convert it to another digit. We used the full MNIST dataset for training.

# Unsafe Images
We used three datasets of real-world harmful images to study
the practical application of counterfactual subobject expla-
nations. 

<li> First, we used the NSFW images dataset [Ala-
giri, 2021] consisting of 334,327 images by selecting the
“porn” and “neutral” classes.


<li> Second, we used a cyberbul-
lying images [Vishwamitra et al., 2021] dataset consisting of
nearly 20,000 images belonging to the classes “cyberbully-
ing” and “non-cyberbullying”. 


<li> Third, we used a self-harm
images dataset, which was scraped from posts under self-
harm-related tags on Tumblr, consisting of 5000 images with
classes “self-harm” and “non self-harm”. A counterfactual
explanation based on subobject regions is expected to iden-
tify the subobject regions that caused the harm in such im-
ages and mask such regions to render them safe. We trained
separate binary classifiers on the three datasets. The self-harm dataset is under IRB review.