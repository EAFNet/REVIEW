import torch
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input)    # 将输入保存起来，在backward时使用
        #output = input-torch.exp(-input)+1
        output=torch.where(input<0.,torch.exp(input)-1,input-torch.exp(-input)+1)


        return output
    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu
        input_,  = ctx.saved_tensors
        grad_input1 = grad_output.clone() #**就是上面推导的 σ（l+1）**

        grad_input=torch.where(input_<0.,grad_input1*torch.exp(input_) ,grad_input1*((torch.exp(-input_)+1)))
        return grad_input
def myrelu(input):
    # MyReLU()是创建一个MyReLU对象，
    # Function类利用了Python __call__操作，使得可以直接使用对象调用__call__制定的方法
    # __call__指定的方法是forward，因此下面这句MyReLU（）（input_）相当于
    # return MyReLU().forward(input_)
    return MyReLU().apply(input)




