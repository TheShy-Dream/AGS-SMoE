import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1) #[K,D]
            inputs_size = inputs.size() #[B,H,W,D]
            inputs_flatten = inputs.view(-1, embedding_size) #准备嵌入[B*H*W,D]

            codebook_sqr = torch.sum(codebook ** 2, dim=1) #每个嵌入向量的长度计算 [K]
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True) #inputs向量的长度计算 [B*H*W,1]

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0) #[B*H*W,K]

            '''
            output=α×mat1@mat2+β×input
            mat1@mat2就是input和codebook的相似度 [B*H*W,K]样本对应codebook的各向量的相似度
            input 就是 codebook_sqr + inputs_sqr [B*H*W,K]每一行对应一个input通道向量和各codebook的向量的长度综合
            两个矩阵中的欧几里得距离需要逐向量进行平方差求和，
            可以看作是dis(A,B)=$\sqrt{\sum((n-m)^2)}$两个向量的L2范数
            平方打开N^2+M^2-2*N*M就是上式咯
            '''

            _, indices_flatten = torch.min(distances, dim=1) #(min,min_indices)拿的是索引咯，那就不可微了 [B*H*W]
            indices = indices_flatten.view(*inputs_size[:-1]) #[B*H*W]->[B,H,W]
            ctx.mark_non_differentiable(indices) #标记不可微

            return indices #针对每个向量返回索引，可能可以用于可视化[B,H,W]

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function): #这里直接通过indice来索引了，但是这个参数显然是不可微的。所以需要straight through
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook) #[B,H,W]
        indices_flatten = indices.view(-1) #[B*H*W]
        ctx.save_for_backward(indices_flatten, codebook) #存储张量用于反向传播
        ctx.mark_non_differentiable(indices_flatten) #标记不可微

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten) #[B*H*W,D]
        codes = codes_flatten.view_as(inputs) #[B,H,W,D]

        return (codes, indices_flatten) #[B,H,W,D]，[B*H*W]

    @staticmethod
    def backward(ctx, grad_output, grad_indices): #[B,H,W,D] [K,D]
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]: #这里是input 根据前向传播 跳过了decoder的输出梯度直接给到encoder不经过codebook
            # Straight-through estimator
            grad_inputs = grad_output.clone() #对于input不计算梯度，直接把输入的梯度原封不懂输出
        if ctx.needs_input_grad[1]: #这个是codebook 其实不算
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors #[B*H*W],[K,D]
            embedding_size = codebook.size(1) #D

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))#[B*H*W,D]
            grad_codebook = torch.zeros_like(codebook)#[K,D]
            grad_codebook.index_add_(0, indices, grad_output_flatten) #[K,D],就是input对于codebook的梯度分配

        return (grad_inputs, grad_codebook) #[B,H,W,D] [K,D]

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
