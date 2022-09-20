# Some Intuitions in Linear Algebra


<!--more-->

## 基本概念
- 线性变换/映射：不变：直线、直线比例、原点。 
- 仿射变换/映射：线性变换+平移。高维线性变换的投影。 
- 行列式：为了解线性方程组定义。-> Cramer 法则。 线性变换的伸缩因子（小于零时翻面）。 
- 矩阵：运动（旋转、投影、拉伸）。指定基下的线性变换。列为新基。 
- 特征值：运动速度。相似不变量：和为迹，积为行列式。 
- 特征向量：运动方向。 
- 特征值分解：$A = PBP^{-1}$  以特征向量为基，B为对角矩阵。P旋转，B拉伸。 
- 正交矩阵：（向量长度和夹角不变，即两点欧式距离不变）旋转与镜射。行列式为1时为旋转矩阵，-1时为瑕旋转矩阵（旋转+镜射） 
- 对称矩阵：二次型。（行列空间相同） 
- 对角矩阵：拉伸。 
- 相似矩阵：同一线性变换在不同基下的表示。换个视角看。 


## Jacobian矩阵 

$$ c = \pm\sqrt{a^2 + b^2} $$

$$J \equiv \left[\begin{array}{ccc}
\frac{\partial f }{\partial x\_{1}} & \cdots & \frac{\partial f }{\partial x\_{n}}
\end{array}\right]=\left[\begin{array}{ccc}
\frac{\partial f\_{1}}{\partial x\_{1}} & \cdots & \frac{\partial f\_{1}}{\partial x\_{n}} \\\\ 
\vdots & \ddots & \vdots \\\\ 
\frac{\partial f\_{m}}{\partial x\_{1}} & \cdots & \frac{\partial f\_{m}}{\partial x\_{n}}
\end{array}\right]$$


- 以一阶偏导数排列。 
- 体现了可微方程在给出点的最优线性逼近，本质是导数，即微分映射的坐标表示。 
- 导数\微分是切空间上的线性变换，以一组基底（欧式空间中，基底选择自然映射构建即可）给出切空间上点的坐标，线性变换就具体化为一个矩阵。欧式空间中，这个矩阵就是雅可比矩阵。 
- Jacobian行列式几何意义：矩阵对应的线性变换前后的面积比。故而积分中变换坐标时会乘以Jacobian行列式。 


## Hessian矩阵 

$$H =\left[\begin{array}{cccc}
\frac{\partial^{2} f}{\partial x\_{1}^{2}} & \frac{\partial^{2} f}{\partial x\_{1} \partial x\_{2}} & \cdots & \frac{\partial^{2} f}{\partial x\_{1} \partial x\_{n}} \\\\ 
\frac{\partial^{2} f}{\partial x\_{2} \partial x\_{1}} & \frac{\partial^{2} f}{\partial x\_{2}^{2}} & \cdots & \frac{\partial^{2} f}{\partial x\_{2} \partial x\_{n}} \\\\ 
\vdots & \vdots & \ddots & \vdots \\\\ 
\frac{\partial^{2} f}{\partial x\_{n} \partial x\_{1}} & \frac{\partial^{2} f}{\partial x\_{n} \partial x\_{2}} & \cdots & \frac{\partial^{2} f}{\partial x\_{n}^{2}}
\end{array}\right]$$

- 多元函数的二阶偏导数构成的方阵，描述了函数的局部曲率。 
- 可判断多元函数的极值：正定极小、负定极大、不定鞍点。 
- 其特征值反应该点特征向量方向的凹凸性，特征值越大，凸性越强。最大特征值和对应特征向量反应其邻域二维曲线最大曲率强度和方向。 
- 梯度的Jacobian即为Hessian。 

> 向量对向量求导： 
> - 一阶导：Jacobian 
> - 二阶导：Hessian 
 
