# 一文彻底搞懂BP算法：原理推导+数据演示+项目实战

反向传播算法（Backpropagation Algorithm，简称BP算法）是深度学习的重要思想基础，对于初学者来说也是必须要掌握的基础知识！本文希望以一个清晰的脉络和详细的说明，来让读者彻底明白BP算法的原理和计算过程。

全文分为上下两篇，上篇主要介绍BP算法的原理（即公式的推导），介绍完原理之后，我们会将一些具体的数据带入一个简单的三层神经网络中，去完整的体验一遍BP算法的计算过程；下篇是一个项目实战，我们将带着读者一起亲手实现一个BP神经网络（不使用任何第三方的深度学习框架）来解决一个具体的问题。

# 1\. BP算法的推导

![图1 一个简单的三层神经网络](https://upload-images.jianshu.io/upload_images/13056713-8fa1e0378e60af5b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



图 1 所示是一个简单的三层（两个隐藏层，一个输出层）神经网络结构，假设我们使用这个神经网络来解决二分类问题，我们给这个网络一个输入样本$(x_1,x_2)$，通过前向运算得到输出$\hat{y}$。输出值$\hat{y}$的值域为$[0,1]$，例如$\hat{y}$的值越接近0，代表该样本是"0"类的可能性越大，反之是"1"类的可能性大。

##1.1 前向传播的计算

为了便于理解后续的内容，我们需要先搞清楚前向传播的计算过程，以图1所示的内容为例：

输入的样本为：

$$
\overrightarrow{\mathrm{a}}=\left(x_{1}, x_{2}\right)
$$

第一层网络的参数为：

$$
W^{(1)}=\left[\begin{array}{l}{W_{\left(x_{1}, 1\right)}, W_{\left(x_{2}, 1\right)}} \\ {W_{\left(x_{1}, 2\right)}, W_{\left(x_{2}, 2\right)}} \\ {W_{\left(x_{1}, 3\right)}, W_{\left(x_{2}, 3\right)}}\end{array}\right], \qquad b^{(1)}=\left[b_{1}, b_{2}, b_{3}\right]
$$

第二层网络的参数为：

$$
W^{(2)}=\left[\begin{array}{l}{W_{(1,4)}, W_{(2,4)}, W_{(3,4)}} \\ {W_{(1,5)}, W_{(2,5)}, w_{(3,5)}}\end{array}\right], \qquad b^{(2)}=\left[b_{4}, b_{5}\right]
$$

第三层网络的参数为：

$$
W^{3}=\left[w_{(4,6)}, w_{(5,6)}\right], \quad b^{(3)}=\left[b_{6}\right]
$$

### 1.1.1 第一层隐藏层的计算

![图2 计算第一层隐藏层](https://upload-images.jianshu.io/upload_images/13056713-e076570b8b21eb63?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


第一层隐藏层有三个神经元：$neu_1$、$neu_2$和$neu_3$。该层的输入为：

$$
Z^{(1)}=W^{(1)} *(\vec{a})^{T}+\left(b^{(1)}\right)^{T}
$$

以$neu_1$神经元为例，则其输入为：

$$
z_{1}=w_{\left(x_{1}, 1\right)} * x_{1}+w_{\left(x_{2}, 1\right)} * x_{2}+b_{1}
$$

同理有：

$$
\begin{array}{l}{z_{2}=w_{\left(x_{1}, 2\right)} * x_{1}+w_{\left(x_{2}, 2\right)} * x_{2}+b_{2}} \\ {z_{3}=w_{\left(x_{1}, 3\right)} * x_{1}+w_{\left(x_{2}, 3\right)} * x_{2}+b_{3}}\end{array}
$$

假设我们选择函数$f(x)$作为该层的激活函数（图1中的激活函数都标了一个下标，一般情况下，同一层的激活函数都是一样的，不同层可以选择不同的激活函数），那么该层的输出为：$f_1(z_1)$、$f_2(z_2)$和$f_3(z_3)$。

### 1.1.2 第二层隐藏层的计算

![图3 计算第二层隐藏层](https://upload-images.jianshu.io/upload_images/13056713-684285514d2a07da?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


第二层隐藏层有两个神经元：$neu_4$和$neu_5$。该层的输入为：

$$
\mathbf{z}^{(2)}=\boldsymbol{W}^{(2)} *\left[f_{1}\left(z_{1}\right), f_{2}\left(z_{2}\right), f_{3}\left(z_{3}\right)\right]^{T}+\left(\boldsymbol{b}^{(2)}\right)^{T}
$$

即第二层的输入是第一层的输出乘以第二层的权重，再加上第二层的偏置。因此得到和的输入分别为：

$$
\begin{array}{l}{z_{4}=w_{(1,4)} * f_{1}\left(z_{1}\right)+w_{(2,4)} * f_{2}\left(z_{2}\right)+w_{(3,4)} * f_{3}\left(z_{3}\right)+b_{4}} \\ {z_{5}=w_{(1,5)} * f_{1}\left(z_{1}\right)+w_{(2,5)} * f_{2}\left(z_{2}\right)+w_{(3,5)} * f_{3}\left(z_{3}\right)+b_{5}}\end{array}
$$

该层的输出分别为：$f_4(z_4)$和$f_5(z_5)$。

### 1.1.3 输出层的计算

![图4 计算输出层](https://upload-images.jianshu.io/upload_images/13056713-ebb295e1f681e589?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

输出层只有一个神经元$neu_6$：。该层的输入为：

$$
\mathbf{z}^{(3)}=\boldsymbol{W}^{(3)} *\left[f_{4}\left(z_{4}\right), f_{5}\left(z_{5}\right)\right]^{T}+\left(\boldsymbol{b}^{(3)}\right)^{T}
$$

即：

$$
z_{6}=w_{(4,6)} * f_{4}\left(z_{4}\right)+w_{(5,6)} * f_{5}\left(z_{5}\right)+b_{6}
$$

因为该网络要解决的是一个二分类问题，所以输出层的激活函数也可以使用一个Sigmoid型函数，神经网络最后的输出为：$f_6(z_6)$。

## 1.2 反向传播的计算

在1.1节里，我们已经了解了数据沿着神经网络前向传播的过程，这一节我们来介绍更重要的反向传播的计算过程。假设我们使用随机梯度下降的方式来学习神经网络的参数，损失函数定义为$\mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})$，其中 $\mathrm{y}$ 是该样本的真实类标。使用梯度下降进行参数的学习，我们必须计算出损失函数关于神经网络中各层参数（权重$w$和偏置$b$）的偏导数。

![](https://upload-images.jianshu.io/upload_images/13056713-07f069731997296c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设我们要对第$k$层隐藏层的参数$W^{(k)}$和$b^{(k)}$求偏导数，即求 $\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial W^{(k)}}$和 $\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial b^{(k)}}$。假设$Z^{(k)}$代表第$k$层神经元的输入，即$z^{(k)}=W^{(k)} * n^{(k-1)}+b^{(k)}$，其中$n^{(k-1)}$ 为前一层神经元的输出，则根据链式法则有：

$$\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial W^{(k)}}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial \mathrm{z}^{(k)}} * \frac{\partial z^{(k)}}{\partial W^{(k)}}$$

$$\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial b^{(k)}}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial z^{(k)}} * \frac{\partial z^{(k)}}{\partial b^{(k)}}$$

因此，我们只需要计算偏导数$\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial z^{(k)}}$、$\frac{\partial z^{(k)}}{\partial W^{(k)}}$ 和 $\frac{\partial z^{(k)}}{\partial b^{(k)}}$。




### 1.2.1 计算偏导数$\frac{\partial z^{(k)}}{\partial W^{(k)}}$

前面说过，第k层神经元的输入为：$z^{(k)}=W^{(k)} * n^{(k-1)}+b^{(k)}$，因此可以得到：

![](https://upload-images.jianshu.io/upload_images/13056713-d82ff15d292c8671?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上式中，$W_{m :}^{(k)}$代表第$k$层神经元的权重矩阵$W^{(k)}$的第$m$行，$W_{mn}^{(k)}$代表第$k$层神经元的权重矩阵$W^{(k)}$的第$m$行中的第$n$列。

我们以1.1节中的简单神经网络为例，假设我们要计算第一层隐藏层的神经元关于权重矩阵的导数，则有：

$$\frac{\partial z^{(1)}}{\partial W^{(1)}}=\left(x_{1}, x_{2}\right)^{T}=\left(\begin{array}{c}{x_{1}} \\ {x_{2}}\end{array}\right)$$

### 1.2.2 计算偏导数$\frac{\partial z^{(k)}}{\partial b^{(k)}}$


因为偏置b是一个常数项，因此偏导数的计算也很简单：

$$
\frac{\partial z^{(k)}}{\partial b^{(k)}}=\left[\begin{array}{ccc}{\frac{\partial\left(W_{1 :}^{(k)} * n^{(k-1)}+b_{1}\right)}{\partial b_{1}}} & {\cdots} & {\frac{\partial\left(W_{1 :}^{(k)} * n^{(k-1)}+b_{1}\right)}{\partial b_{m}}} \\ {\frac{\partial\left(W_{m :}(k) * n^{(k-1)}+b_{m}\right)}{\partial b_{1}}} & {\dots} & {\frac{\partial\left(W_{m_{i}^{(k)}} * n^{(k-1)}+b_{m}\right)}{\partial b_{m}}}\end{array}\right]
$$

依然以第一层隐藏层的神经元为例，则有：

$$
\frac{\partial z^{(1)}}{\partial b^{(1)}}=\left[\begin{array}{lll}{1} & {0} & {0} \\ {0} & {1} & {0} \\ {0} & {0} & {1}\end{array}\right]
$$

### 1.2.3 计算偏导数$\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial z^{(k)}}$

偏导数$\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial z^{(k)}}$又称为**误差项（error term，也称为“灵敏度”）**，一般用$\delta$表示，例如$\delta^{(1)}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{y})}{\partial z^{(1)}}$是第一层神经元的误差项，其值的大小代表了第一层神经元对于最终总误差的影响大小。


根据第一节的前向计算，我们知道第$k+1$层的输入与第$k$层的输出之间的关系为：

$$
z^{(k+1)}=W^{(k+1)} * n^{(k)}+b^{k+1}
$$

又因为$n^{(k)}=f_{k}\left(z^{(k)}\right)$，根据链式法则，我们可以得到$\delta(k)$为：

$$
\begin{aligned} & \delta^{(k)}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial z^{(k)}} \\=& \frac{\partial n^{(k)}}{\partial z^{(k)}} * \frac{\partial z^{(k+1)}}{\partial n^{(k)}} * \frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial z^{(k+1)}} \\=& \frac{\partial n^{(k)}}{\partial z^{(k)}} * \frac{\partial z^{(k+1)}}{\partial n^{(k)}} * \delta^{(k+1)} \\=& f_{k}^{\prime}\left(z^{(k)}\right) *\left(\left(W^{(k+1)}\right)^{T} * \delta^{(k+1)}\right) \end{aligned}
$$

由上式我们可以看到，第$k$层神经元的误差项$\delta^{(k)}$是由第$k+1$层的误差项乘以第$k+1$层的权重，再乘以第$k$层激活函数的导数（梯度）得到的。这就是误差的反向传播。
	现在我们已经计算出了偏导数 $\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial z^{(k)}}$、$\frac{\partial z^{(k)}}{\partial W^{(k)}}$和 $\frac{\partial z^{(k)}}{\partial b^{(k)}}$，则$\frac{\partial \mathrm{L}(\mathrm{y} , \widehat{\mathrm{y}})}{\partial \mathrm{W}^{(k)}}$和 $\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial b^{(k)}}$可分别表示为：

$$
\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{y})}{\partial W^{(k)}}=\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{y})}{\partial z^{(\mathrm{k})}} * \frac{\partial z^{(k)}}{\partial W^{(k)}}=\delta^{(k)} *\left(n^{(k-1)}\right)^{T}
$$


$$
\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial b^{(k)}}=\frac{\partial \mathrm{L}(\mathrm{y}, \widehat{\mathrm{y}})}{\partial z^{(k)}} * \frac{\partial z^{(k)}}{\partial b^{(k)}}=\delta^{(k)}
$$


下面是基于随机梯度下降更新参数的反向传播算法：

![](https://upload-images.jianshu.io/upload_images/13056713-19c09dbf953a646d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**单纯的公式推导看起来有些枯燥，下面我们将实际的数据带入图1所示的神经网络中，完整的计算一遍。**

# 2\. 图解BP算法

![图5 图解BP算法](https://upload-images.jianshu.io/upload_images/13056713-553dca76ddfce469?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们依然使用如图5所示的简单的神经网络，其中所有参数的初始值如下：

输入的样本为（假设其真实类标为"1"）：

$$
\overrightarrow{\mathrm{a}}=\left(x_{1}, x_{2}\right)=(1,2)
$$

第一层网络的参数为：

$$
W^{(1)}=\left[\begin{array}{c}{w_{\left(x_{1}, 1\right)}, w_{\left(x_{2}, 1\right)}} \\ {w_{\left(x_{1}, 2\right)}, w_{\left(x_{2}, 2\right)}} \\ {w_{\left(x_{1}, 3\right)}, w_{\left(x_{2}, 3\right)}}\end{array}\right]=\left[\begin{array}{cc}{2} & {1} \\ {1} & {3} \\ {3} & {2}\end{array}\right], \quad b^{(1)}=\left[b_{1}, b_{2}, b_{3}\right]^{T}=[1,2,3]^{T}
$$

第二层网络的参数为：

$$
W^{(2)}=\left[\begin{array}{c}{W_{(1,4)}, W_{(2,4)}, W_{(3,4)}} \\ {w_{(1,5)}, W_{(2,5)}, w_{(3,5)}}\end{array}\right]=\left[\begin{array}{ccc}{1} & {1} & {2} \\ {3} & {2} & {2}\end{array}\right], \quad b^{(2)}=\left[b_{4}, b_{5}\right]^{T}=[2,1]^{T}
$$

第三层网络的参数为：

$$
W^{3}=\left[w_{(4,6)}, w_{(5,6)}\right]=[1,3], \quad b^{(3)}=\left[b_{6}\right]=[2]
$$

假设所有的激活函数均为Logistic函数：$f^{(k)}(x)=\frac{1}{1+e^{-x}}$。使用均方误差函数作为损失函数：

$$\mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})=\mathrm{E}(\mathrm{y}-\hat{\mathrm{y}})^{2}$$

为了方便求导，我们将损失函数简化为：

$$
\mathrm{L}(\mathrm{y}, \hat{y})=\frac{1}{2} \sum(\mathrm{y}-\hat{\mathrm{y}})^{2}
$$

## 2.1 前向传播

我们首先初始化神经网络的参数，计算第一层神经元：

![](https://upload-images.jianshu.io/upload_images/13056713-f8fc7ad345ad6e5c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$
\begin{array}{c}{z_{1}=w_{\left(x_{1}, 1\right)} * x_{1}+w_{\left(x_{2}, 1\right)} * x_{2}+b_{1}} \\ {=2 * 1+1 * 2+1 } \\ {=5}\end{array}
$$

$$f_{1}\left(z_{1}\right)=\frac{1}{1+e^{-z_{1}}}=0.993307149075715$$

上图中我们计算出了第一层隐藏层的第一个神经元的输入$z_1$和输出$f_1(z_1)$，同理可以计算第二个和第三个神经元的输入和输出：

![](https://upload-images.jianshu.io/upload_images/13056713-b29719e65a7e36fe?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$
\begin{array}{c}{z_{2}=w_{\left(x_{1}, 2\right)} * x_{1}+w_{\left(x_{2}, 2\right)} * x_{2}+b_{2}} \\ {=1 * 1+3 * 2+2=9}\end{array}
$$

$$
f_{2}\left(z_{2}\right)=\frac{1}{1+e^{-z_{2}}}=0.999876605424014
$$

$$
\begin{array}{c}{z_{3}=w_{\left(x_{1}, 3\right)} * x_{1}+w_{\left(x_{2}, 3\right)} * x_{2}+b_{3}} \\ {=3 * 1+2 * 2+3=10}\end{array}
$$

$$
f_{3}\left(z_{3}\right)=\frac{1}{1+e^{-z_{3}}}=0.999954602131298
$$

接下来是第二层隐藏层的计算，首先我们计算第二层的第一个神经元的输入z₄和输出f₄(z₄)：

![](https://upload-images.jianshu.io/upload_images/13056713-2768830f95d1809d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$
\begin{array}{c}{z_{4}=w_{(1,4)} * f_{1}\left(z_{1}\right)+w_{(2,4)} * f_{2}\left(z_{2}\right)+w_{(3,4)} * f_{3}\left(z_{3}\right)+b_{4}} \\ {=1 * 0.993307149075715+1 * 0.999876605424014+2 * 0.999954602131298+2} \\ {=5.993092958762325}\end{array}
$$

$$
f_{4}\left(z_{4}\right)=\frac{1}{1+e^{-z_{4}}}=0.997510281884102
$$

同样方法可以计算该层的第二个神经元的输入$z_5$和输出$f_5(z_5)$：

![](https://upload-images.jianshu.io/upload_images/13056713-2f0f27425a866d00?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$
\begin{array}{c}{z_{5}=w_{(1,5)} * f_{1}\left(z_{1}\right)+w_{(2,5)} * f_{2}\left(z_{2}\right)+w_{(3,5)} * f_{3}\left(z_{3}\right)+b_{5}} \\ {=3 * 0.993307149075715+3 * 0.999876605424014+2 * 0.999954602131298+1} \\ {=8.979460467761783}\end{array}
$$

$$
f_{5}\left(z_{5}\right)=\frac{1}{1+e^{-z_{5}}}=0.999874045072167
$$

最后计算输出层的输入$z_6$和输出$f_6(z_6)$：

![](https://upload-images.jianshu.io/upload_images/13056713-8b0e94d6b5a02ef6?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

$$
\begin{array}{c}{z_{6}=w_{(4,6)} * f_{4}\left(z_{4}\right)+w_{(5,6)} * f_{5}\left(z_{5}\right)+b_{6}} \\ {=1 * 0.997510281884102+3 * 0.999874045072167+2} \\ {=5.997132417100603}\end{array}
$$

$$
f_{6}\left(z_{6}\right)=\frac{1}{1+e^{-z_{6}}}=0.997520293823002
$$


## 2.2 误差反向传播

![](https://upload-images.jianshu.io/upload_images/13056713-106caa21dad2c0c1?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

首先计算输出层的误差项$δ_3$，我们的误差函数为$\mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})=\frac{1}{2} \sum(\mathrm{y}-\hat{\mathrm{y}})^{2}$，由于该样本的类标为“1”，而预测值为$0.997520293823002$，因此误差为$0.002479706176998$，输出层的误差项为：

$$
\begin{array}{c}{\delta_{3}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial z^{(3)}}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial n^{(3)}} * \frac{\partial n^{(3)}}{\partial^{(3)}}=[-0.002479706176998] * f^{(3)^{\prime}}\left(\mathrm{z}^{3}\right)} \\ {=[0.002473557234274] *[-0.002479706176998]} \\ {=[-0.000006133695153]}\end{array}
$$


接着计算第二层隐藏层的误差项，根据误差项的计算公式有：

$$
\delta^{(2)}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{\mathrm{y}})}{\partial z^{(2)}}=f^{(2)^{\prime}}\left(z^{(2)}\right) *\left(\left(W^{(3)}\right)^{T} * \delta^{(3)}\right)
$$
$$
=\begin{bmatrix}{f_{4}^{\prime}\left(z_{4}\right)} & {0} \\ {0} & {f_{5}^{\prime}\left(z_{5}\right)}\end{bmatrix} *\left(\left[\begin{array}{l}{1} \\ {3}\end{array}\right] *[-0.000006133695153]\right)
$$
$$
=\left[\begin{array}{cc}{0.002483519419601} & {0} \\ {0} & {0.000125939063189}\end{array}\right] *\left[\begin{array}{l}{-0.000006133695153} \\ {-0.000018401085459}\end{array}\right]
$$
$$
=\begin{bmatrix}{-0.000000015233151} \\ {-0.000000002317415}\end{bmatrix}
$$


最后是计算第一层隐藏层的误差项：

$$
\begin{aligned} \delta^{(1)}=\frac{\partial \mathrm{L}(\mathrm{y}, \hat{y})}{\partial n^{(1)}} &=f^{(1)^{\prime}}\left(n^{(1)}\right) *\left(\left(W^{(2)}\right)^{T} * \delta^{(2)}\right) \\ &=\left[\begin{array}{ccc}{f_{1}^{\prime}\left(z_{1}\right)} & {0} & {0} \\ {0} & {f_{2}^{\prime}\left(z_{2}\right)} & {0} \\ {0} & {0} & {f_{3}^{\prime}\left(z_{3}\right)}\end{array}\right] \end{aligned}
$$
$$
=\begin{bmatrix}{0.006648056670790} & {0} & {0} \\ {0} & {0.000123379349765} & {0} \\ {0} & {0} & {0.000045395807355}\end{bmatrix}
$$
$$
*\left(\left[\begin{array}{lll}{1} & {1} & {2} \\ {3} & {2} & {2}\end{array}\right]^{T} *\left[\begin{array}{c}{-0.000000015233151} \\ {-0.000000002317415}\end{array}\right]\right)
$$
$$
=\left[\begin{array}{l}{-0.000000000147490} \\ {-0.000000000002451} \\ {-0.000000000001593}\end{array}\right]
$$

## 2.3 更新参数

上一小节中我们已经计算出了每一层的误差项，现在我们要利用每一层的误差项和梯度来更新每一层的参数，权重W和偏置b的更新公式如下：

$$
\begin{array}{c}{W^{(k)}=W^{(k)}-\alpha\left(\delta^{(k)}\left(n^{(k-1)}\right)^{T}+W^{(k)}\right)} \\ {b^{(k)}=b^{(k)}-\alpha \delta^{(k)}}\end{array}
$$

通常权重$W$的更新会加上一个正则化项来避免过拟合，这里为了简化计算，我们省去了正则化项。上式中$\alpha$的是学习率，我们设其值为0.1。参数更新的计算相对简单，每一层的计算方式都相同，因此本文仅演示第一层隐藏层的参数更新：

![](https://upload-images.jianshu.io/upload_images/13056713-1845ea8fb509679c?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 3\. 小结

至此，我们已经完整介绍了BP算法的原理，并使用具体的数值做了计算。下面我们将带着读者一起亲手实现一个BP神经网络（不使用任何第三方的深度学习框架）。


我们将自己实现BP算法（不使用第三方的算法框架），并用来解决鸢尾花分类问题。

![图1 鸢尾花](https://upload-images.jianshu.io/upload_images/13056713-f8db8fb1a15c3442?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



鸢尾花数据集如图2所示，总共有三个品种的鸢尾花（setosa、versicolor和virginica），每个类别50条样本数据，每个样本有四个特征（花萼长度、花萼宽度、花瓣长度以及花瓣宽度）。

![图2 鸢尾花数据集](https://upload-images.jianshu.io/upload_images/13056713-05f644b8ac047aca?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



首先我们导入需要的包：


```
from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import math
```


接下来我们实现一个数据集的加载和预处理的函数`load_dataset`：

```
def load_dataset(dataset_path, n_train_data):
    """加载数据集，对数据进行预处理，并划分训练集和验证集
    :param dataset_path: 数据集文件路径
    :param n_train_data: 训练集的数据量
    :return: 划分好的训练集和验证集
    """
    dataset = []
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(dataset_path, 'r') as file:
        # 读取CSV文件，以逗号为分隔符
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            # 将字符串类型的特征值转换为浮点型
            row[0:4] = list(map(float, row[0:4]))
            # 将标签替换为整型
            row[4] = label_dict[row[4]]
            # 将处理好的数据加入数据集中
            dataset.append(row)

    # 对数据进行归一化处理
    dataset = np.array(dataset)
    mms = MinMaxScaler()
    for i in range(dataset.shape[1] - 1):
        dataset[:, i] = mms.fit_transform(dataset[:, i].reshape(-1, 1)).flatten()

    # 将类标转为整型
    dataset = dataset.tolist()
    for row in dataset:
        row[4] = int(row[4])
    # 打乱数据集
    random.shuffle(dataset)

    # 划分训练集和验证集
    train_data = dataset[0:n_train_data]
    val_data = dataset[n_train_data:]

    return train_data, val_data
```

在`load_dataset`函数中，我们实现了数据集的读取、数据的归一化处理以及对数据集进行了`shuffle`操作等，最后函数返回了划分好的训练集和验证集。

实现数据预处理之后，接下来我们开始实现BP算法的关键部分（**如果读者对算法原理有不清楚的地方，可以查看"一文彻底搞懂BP算法：原理推导+数据演示+项目实战（上篇）"**）。首先我们实现神经元的计算部分、激活函数以及激活函数的求导部分。


```
def fun_z(weights, inputs):
    """计算神经元的输入：z = weight * inputs + b
    :param weights: 网络参数（权重矩阵和偏置项）
    :param inputs: 上一层神经元的输出
    :return: 当前层神经元的输入
    """
    bias_term = weights[-1]
    z = 0
    for i in range(len(weights)-1):
        z += weights[i] * inputs[i]
    z += bias_term
    return z


def sigmoid(z):
    """激活函数(Sigmoid)：f(z) = Sigmoid(z)
    :param z: 神经元的输入
    :return: 神经元的输出
    """
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(output):
    """Sigmoid激活函数求导
    :param output: 激活函数的输出值
    :return: 求导计算结果
    """
    return output * (1.0 - output)
```


函数`fun_z`实现了公式"z = weight * inputs + b"，其中inputs是上一层网络的输出，weight是当前层的权重矩阵，b是当前层的偏置项，计算得到的z是当前层的输入。

函数`sigmoid`是Sigmoid激活函数的实现，将z作为激活函数的输入，计算得到当前层的输出，并传递到下一层。

函数`sigmoid_derivative`是Sigmoid函数求导的实现，在误差反向传播的时候需要用到。

接下来我们实现BP网络的前向传播：


```
def forward_propagate(network, inputs):
    """前向传播计算
    :param network: 神经网络
    :param inputs: 一个样本数据
    :return: 前向传播计算的结果
    """
    for layer in network:  # 循环计算每一层
        new_inputs = []
        for neuron in layer:  # 循环计算每一层的每一个神经元
            z = fun_z(neuron['weights'], inputs)
            neuron['output'] = sigmoid(z)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
```


前向计算的过程比较简单，和我们在上篇中介绍的计算过程一致。稍微麻烦一点的是误差反向传播的计算：


```
def backward_propagate_error(network, actual_label):
    """误差进行反向传播
    :param network: 神经网络
    :param actual_label: 真实的标签值
    :return:
    """
    for i in reversed(range(len(network))):  # 从最后一层开始计算误差
        layer = network[i]
        errors = list()
        if i != len(network)-1:  # 不是输出层
            for j in range(len(layer)):  # 计算每一个神经元的误差
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:  # 输出层
            for j in range(len(layer)):  # 计算每一个神经元的误差
                neuron = layer[j]
                errors.append(actual_label[j] - neuron['output'])
        # 计算误差项 delta
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
```


误差反向传播过程中，我们首先需要根据模型的输出来计算得到误差，然后计算输出层的误差项。得到输出层的误差项之后，我们就可以根据上篇中介绍的"第k层神经元的误差项是由第k+1层的误差项乘以第k+1层的权重，再乘以第k层激活函数的导数得到"来计算其它层的误差项。

在计算得到每一层的误差项之后，我们根据上篇中介绍的权重矩阵和偏置项的更新公式来更新参数：


```
def update_parameters(network, row, l_rate):
    """利用误差更新神经网络的参数（权重矩阵和偏置项）
    :param network: 神经网络
    :param row: 一个样本数据
    :param l_rate: 学习率
    :return:
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:  # 获取上一层网络的输出
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            # 更新权重矩阵
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # 更新偏置项
            neuron['weights'][-1] += l_rate * neuron['delta']
```


到这里所有的关键部分我们都已经实现了，接下来我们实现网络的初始化以及网络的训练部分，首先实现网络的初始化：


```
def initialize_network(n_inputs, n_hidden, n_outputs):
    """初始化BP网络（初始化隐藏层和输出层的参数：权重矩阵和偏置项）
    :param n_inputs: 特征列数
    :param n_hidden: 隐藏层神经元个数
    :param n_outputs: 输出层神经元个数，即分类的总类别数
    :return: 初始化后的神经网络
    """
    network = list()
    # 隐藏层
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # 输出层
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
```


这里我们初始化了一个两层神经网络（一个隐藏层和一个输出层）。在初始化参数的时候，我们将权重矩阵和偏置项放在了一个数组中（"weights"），数组的最后一个元素是偏置项，前面的元素是权重矩阵。

接下来我们实现模型的训练部分：


```
def train(train_data, l_rate, epochs, n_hidden, val_data):
    """训练神经网络（迭代n_epoch个回合）
    :param train_data: 训练集
    :param l_rate: 学习率
    :param epochs: 迭代的回合数
    :param n_hidden: 隐藏层神经元个数
    :param val_data: 验证集
    :return: 训练好的网络
    """
    # 获取特征列数
    n_inputs = len(train_data[0]) - 1
    # 获取分类的总类别数
    n_outputs = len(set([row[-1] for row in train_data]))
    # 初始化网络
    network = initialize_network(n_inputs, n_hidden, n_outputs)

    acc = []
    for epoch in range(epochs):  # 训练epochs个回合
        for row in train_data:
            # 前馈计算
            _ = forward_propagate(network, row)
            # 处理一下类标，用于计算误差
            actual_label = [0 for i in range(n_outputs)]
            actual_label[row[-1]] = 1
            # 误差反向传播计算
            backward_propagate_error(network, actual_label)
            # 更新参数
            update_parameters(network, row, l_rate)
        # 保存当前epoch模型在验证集上的准确率
        acc.append(validation(network, val_data))
    # 绘制出训练过程中模型在验证集上的准确率变化
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(acc)
    plt.show()

    return network
```


我们总共训练了`epochs`个回合，这里我们使用随机梯度下降来优化模型，因此每次都用一个样本来更新参数。接下来我们实现一个函数用来验证模型的效果：


```
def validation(network, val_data):
    """测试模型在验证集上的效果
    :param network: 神经网络
    :param val_data: 验证集
    :return: 模型在验证集上的准确率
    """
    # 获取预测类标
    predicted_label = []
    for row in val_data:
        prediction = predict(network, row)
        predicted_label.append(prediction)
    # 获取真实类标
    actual_label = [row[-1] for row in val_data]
    # 计算准确率
    accuracy = accuracy_calculation(actual_label, predicted_label)
    # print("测试集实际类标：", actual_label)
    # print("测试集上的预测类标：", predicted_label)
    return accuracy
```


训练过程中的每一个回合，我们都用模型对验证集进行一次预测，并将预测的结果保存，用来绘制训练过程中模型在验证集上的准确率的变化过程。准确率的计算以及使用模型进行预测的实现如下：


```
def accuracy_calculation(actual_label, predicted_label):
    """计算准确率
    :param actual_label: 真实类标
    :param predicted_label: 模型预测的类标
    :return: 准确率（百分制）
    """
    correct_count = 0
    for i in range(len(actual_label)):
        if actual_label[i] == predicted_label[i]:
            correct_count += 1
    return correct_count / float(len(actual_label)) * 100.0


def predict(network, row):
    """使用模型对当前输入的数据进行预测
    :param network: 神经网络
    :param row: 一个数据样本
    :return: 预测结果
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
```


最后我们运行代码：


```
if __name__ == "__main__":
    file_path = './iris.csv'

    # 参数设置
    l_rate = 0.2  # 学习率
    epochs = 300  # 迭代训练的次数
    n_hidden = 5  # 隐藏层神经元个数
    n_train_data = 130  # 训练集的大小（总共150条数据，训练集130条，验证集20条）

    # 加载数据并划分训练集和验证集
    train_data, val_data = load_dataset(file_path, n_train_data)
    # 训练模型
    network = train(train_data, l_rate, epochs, n_hidden, val_data)
```


训练过程如图3所示：

![](https://upload-images.jianshu.io/upload_images/13056713-766f8b296b97ab93?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

