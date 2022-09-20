# Quantum Computing Fundamentals


<!--more-->

## Learning Material
- https://quantumalgorithmzoo.org/
- https://qiskit.org//


## 历史
- 量子：一个物理量如果存在最小的不可分割的基本单位，则这个物理量是量子化的，并把最小单位称为量子。
- 量子不是粒子，可以理解为“经典”的反义词。
- 量子力学
	- 等价
		- 波动力学
		- 矩阵力学
	- 基本方程：薛定谔方程
	- 测量影响了粒子本身的状态
- 量子信息科学

![3bf3e8e8d1751bbb7566f64283e5a755.png](/blog/_resources/44d0ebf582164cf2bbba45fee2b9dcbc.png)

## 基础
### 量子力学理论基础
- 希尔伯特空间（Hilbert space）即完备的内积空间，也就是说一个带有内积的完备向量空间
- 量子态可以由Hilbert空间的态矢量表示，量子态的可观测量可以用厄米算符来代表
- 态矢 State Vector（狄拉克符号）
	- 左矢（bra）：$\langle\psi|=\left[\mathrm{c}\_{1}^{\*}, c\_{2}^{\*}, \ldots, c\_{n}^{\*}\right]$
	- 右矢（ket）：$|\psi\rangle=\left[c\_{1}, c\_{2}, \ldots, c\_{n}\right]^{T}$
	- 相同态矢则左右矢互为共轭转置
- 运算
	- 内积：$\langle\alpha \mid \beta\rangle=\sum\_{i=1}^{n} a\_{i}^{*} b\_{i}$
	- 外积：$|\alpha\rangle\langle\beta|=\left[\begin{array}{c}a\_{i} b\_{j}^{*} \\\\ \end{array}\right]\_{n \times n}$
	- 张量积（Kronecker product）：$A \otimes B \equiv\left[\begin{array}{cccc}A\_{11} B & A\_{12} B & \cdots & A\_{1 n} B \\\\ A\_{21} B & A\_{22} B & \cdots & A\_{2 n} B \\\\ \vdots & \vdots & \ddots & \vdots \\\\ A\_{m 1} B & A\_{m 2} B & \cdots & A\_{m n} B\end{array}\right]$
- 能量不同的量子态：
	- 激发态（excited state）：$|e\rangle=\left[\begin{array}{l}1 \\\\ 0\end{array}\right]$
	- 基态（ground state）：$|g\rangle=\left[\begin{array}{l}0 \\\\ 1\end{array}\right]$
- 量子比特（quantum bits）：
	- $|0\rangle = |e\rangle$
	- $|1\rangle = |g\rangle$
	- 任何叠加态都可以写成其线性组合：$|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$
		- $\alpha$、$\beta$称为振幅，满足归一化条件：$|\alpha|^{2}+|\beta|^{2}=1$
- 封闭的(closed)量子系统的演化（evolution）由酉变换（unitary transformation）来描述：$\left|\psi\_{2}\right\rangle=U\left|\psi\_{1}\right\rangle$。即量子态演化本质是矩阵乘法。
	- 酉矩阵：$U U^{*}=I$，正交矩阵的推广，可逆
	- 各种形式的酉矩阵被称作量子门。eg. Puali Matrixes（spin matrices）:
		- $\sigma\_{0} \equiv I \equiv\left[\begin{array}{ll}1 & 0 \\\\ 0 & 1\end{array}\right]$
		- $\sigma\_{1} \equiv \sigma\_{x} \equiv X \equiv\left[\begin{array}{ll}0 & 1 \\\\ 1 & 0\end{array}\right]$ （quantum NOT gate）
		- $\sigma\_{2} \equiv \sigma\_{y} \equiv Y \equiv\left[\begin{array}{cc}0 & -i \\\\ i & 0\end{array}\right]$
		- $\sigma\_{3} \equiv \sigma\_{z} \equiv Z \equiv\left[\begin{array}{cc}1 & 0 \\\\ 0 & -1\end{array}\right]$
- Superposition State And Measurement
	- 任何一个态可以写成基在复数空间的线性组合：$|\psi\rangle=\alpha|0\rangle+\beta e^{i \theta}|1\rangle$
	- 测量：将态投影到另一个态上，概率是其内积的平方：$P\_{\alpha}=|\langle\psi \mid \alpha\rangle|^{2}$
	- 其他概率投影到正交态：$P\_{\alpha \perp}=1-P\_{\alpha}$
- Phase, Pure State and Mixed State
	- 无法通过测量得到相位信息$\theta$，量子态的相位是相干性的表现
	- 纯态：具有概率和相位（量子相干性）的量子态，系统的状态由波函数或态矢量描述
	- 混合态：纯态的概率叠加，失去了（部分或全部）相位信息，系统的状态由密度矩阵描述
- Density Matrix And Bloch Sphere
	- 密度矩阵：
		- 纯态：$\rho=|\psi\rangle\langle\psi|$
		- 混合态：$\rho=\sum\_{i} P\_{i}\left|\psi\_{i}\right\rangle\left\langle\psi\_{i}\right|$
		- $\rho = \rho^2$当仅当纯态
		- 密度矩阵是对波函数和经典概率分布的推广
		- 孤立系统的密度矩阵满足幺正演化方程，开放系统的密度矩阵演化满足量子主方程
		- $b_i$是一组规范正交基，密度矩阵每个元素$\varrho\_{i j}=\left\langle b\_{i}|\rho| b\_{j}\right\rangle=\sum\_{k} w\_{k}\left\langle b\_{i} \mid \psi\_{k}\right\rangle\left\langle\psi\_{k} \mid b\_{j}\right\rangle$
		- 可观测量A的期望为$\langle A\rangle=\sum\_{i} w\_{i}\left\langle\psi\_{i}|A| \psi\_{i}\right\rangle=\sum\_{i}\left\langle b\_{i}|\rho A| b\_{i}\right\rangle=\operatorname{tr}(\rho A)$
		- 密度算符：线性、非负、自伴、迹为1
		- 
	- Bloch sphere ![1ebabacb149e5ebe60398313c1ed4417.png](/blog/_resources/29b2976731b145e3ba04838a3987cc8b.png)
		- 纯态为球面上的点
			- Z坐标衡量概率
		- 混合态为球内的点
			- 最大的混合态是球心，不存在任何叠加性
- 观测量和计算基下的测量
	- 可观测量（类似于位置、动量）由自伴算子（self-adjoint operators）来表征，自伴有时也称为Hermitian
		- 自伴算子：厄米算符（Hermitian operator），等于自己的厄米共轭的算符：$M^{\dagger}=M$
	- 哈密顿量（Hamiltonian）：质量、电荷，作为参数引入系统
	- 测量算子满足完备性方程（completeness equation）：$\sum\_{i} M\_{i}^{\dagger} M\_{i}=I$
	- 测量方式
		- 投影测量（projective measurements）
		- POVM 测量（Positive Operator-Valued Measure）
- 复合系统和联合测量
	- 复合系统：拥有两个或两个以上的量子比特的量子系统
	- 张量积：两个向量空间形成一个更大向量空间的运算
		- 满足：分配律、复数系数交换律
		- 由子系统生成复合系统（Composite system）
	- 纠缠（entanglement）：态$|\psi\rangle \in H\_{1} \otimes H\_{2}$，不存在$|\alpha\rangle \in H\_{1},|\beta\rangle \in H\_{2}$，使得$|\psi\rangle=|\alpha\rangle \otimes|\beta\rangle$。
- 复合系统的状态演化
	- 两能级的量子系统的状态是通过酉变换来实现演化
	- 复合系统量子态的演化：子系统中量子态的酉变换的张量积

### 量子程序
- 量子逻辑门
- 酉变换
	- $\langle\psi|=\left\langle\varphi\right| U ^{\dagger}$
	- $\langle\psi|=\left\langle\varphi\right| U ^{\dagger}$
	- 两个矢量的内积经过同一个酉变换之后保持不变：$\langle\varphi|\psi\rangle=\langle\varphi|U ^{\dagger}U| \psi\rangle$
- 矩阵的指数函数
	- $A^{n}=\operatorname{diag}\left(A\_{11}^{n}, A\_{22}^{n}, A\_{33}^{n} \ldots\right)$
	- $\mathrm{A}^{n}=\mathrm{UD}^{n} \mathrm{U}^{\dagger}$
	- $e^A=Ue^DU^\dagger$
	- 以A为生成元的酉变换：$U(\theta)=e^{-i\theta A}$
		- 以单位矩阵为生成元，构成一种特殊的酉变换：作用与态矢相当于态矢整体乘一个系数，在密度矩阵中该系数会被消去。
		- 该系数称为量子态的整体相位，对系统没有任何影响，因任何测量和操作都无法分辨两个相同的密度矩阵。
- 单量子比特逻辑门
	- Pauli matrices（spin matrix）：
		- $X=\sigma\_{x}=\left(\begin{array}{cc}0 & 1 \\\\ 1 & 0\end{array}\right)$ 非门
		- $Y=\sigma\_{y}=\left(\begin{array}{cc}0 & -i \\\\ i & 0\end{array}\right)$ 绕Y旋转$\pi$度
		- $Z=\sigma\_{z}=\left(\begin{array}{cc}1 & 0 \\\\ 0 & -1\end{array}\right)$ 绕Z旋转$\pi$度
		- 泡利矩阵的线性组合是完备的二维酉变换的生成元：$\mathrm{U}=\mathrm{e}^{-i \theta\left(a \sigma\_{x}+b \sigma\_{y}+c \sigma\_{z}\right)}$
	- Hadamard（H）门：$\mathrm{H}=\frac{1}{\sqrt{2}}\left[\begin{array}{cc}1 & 1 \\\\ 1 & -1\end{array}\right]$
	- Rotation Operators：
		- RX($\theta$)：$R X(\theta)=e^{\frac{-i \theta X}{2}}=\cos \left(\frac{\theta}{2}\right) I-i \sin \left(\frac{\theta}{2}\right) X=\left[\begin{array}{cc}\cos \left(\frac{\theta}{2}\right) & -i \sin \left(\frac{\theta}{2}\right) \\\\ -i \sin \left(\frac{\theta}{2}\right) & \cos \left(\frac{\theta}{2}\right)\end{array}\right]$
		- RY($\theta$)：$R Y(\theta)=e^{\frac{-i \theta Y}{2}}=\cos \left(\frac{\theta}{2}\right) I-i \sin \left(\frac{\theta}{2}\right) Y=\left[\begin{array}{cc}\cos \left(\frac{\theta}{2}\right) & -\sin \left(\frac{\theta}{2}\right) \\\\ \sin \left(\frac{\theta}{2}\right) & \cos \left(\frac{\theta}{2}\right)\end{array}\right]$
		- RZ($\theta$)：$R Z(\theta)=e^{\frac{-i \theta Z}{2}}=\cos \left(\frac{\theta}{2}\right) I-i \sin \left(\frac{\theta}{2}\right) Z=\left[\begin{array}{cc}e^{\frac{-i \theta}{2}} & 0 \\\\ 0 & e^{\frac{i \theta}{2}}\end{array}\right] \sim \left[\begin{array}{cc}1 & \\\\ & e^{i \theta}\end{array}\right]$
		- RX, RY, RZ意味着量子态在布洛赫球上分别绕X, Y, Z旋转$\theta$角度，所以RX、RY带来概率幅的变化，RZ只有相位的变化。这三种操作使量子态在球上自由移动。
- 多量子和比特逻辑门
	- 所有逻辑操作都是酉变换，所以输入、输出比特数量相等
	- $|01\rangle$中0为高位，1为低位，左高右低
	- CNOT门 
		- $C N O T=\left[\begin{array}{cccc}1 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 1 & 0 & 0\end{array}\right]$ ![0a7c5be5883beb468deabf1037ed2d7b.png](/blog/_resources/89401fef998f43a4bfe19f3175c5fe22.png)
		- $C N O T=\left[\begin{array}{cccc}1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\\\ 0 & 0 & 1 & 0\end{array}\right]$ ![82873c46406725d526a2715d3acad834.png](/blog/_resources/9105b64524c64f99a41269e286f414e8.png)
	- CR门：控制相位门（Controlled phase gate）
		- $C R(\theta)=\left[\begin{array}{cccc}1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & e^{i \theta}\end{array}\right]$
	- ISWAP门：交换两个比特的状态
		- 由 $\sigma\_{x} \otimes \sigma\_{x}+\sigma\_{y} \otimes \sigma\_{y}$ 作为生成元生成
		- $i S W A P(\theta)=\left[\begin{array}{cccc}1 & 0 & 0 & 0 \\\\ 0 & \cos (\theta) & -i \sin (\theta) & 0 \\\\ 0 & -i \sin (\theta) & \cos (\theta) & 0 \\\\ 0 & 0 & 0 & 1\end{array}\right]$
- 量子线路与测量操作
- 量子计算的if和while
	- 基于测量的跳转
	- 基于量子信息的if和while
- 量子逻辑门知识图谱：![921d0d390158641d6144e487880d66f7.png](/blog/_resources/054c05e24c854fd89f4b2771b32caa7c.png)

