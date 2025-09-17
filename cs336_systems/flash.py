import torch

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 获取维度信息
        batch_size, n_queries, d_model = Q.shape
        _, n_keys, _ = K.shape
        # 定义 tile 的大小，作业中要求至少是 16x16
        B_q = 64
        B_k = 64
        # 计算分块数量
        T_q = (n_queries + B_q - 1) // B_q
        T_k = (n_keys + B_k - 1) // B_k
        # 初始化输出 O 和 log-sum-exp L
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, n_queries, device=Q.device)

        scale = d_model ** -0.5

        # Algorithm 1 的 pytorch 实现
        # 遍历 Query tile
        for i in range(T_q):
            q_start, q_end = i * B_q, min((i + 1) * B_q, n_queries)
            Q_i = Q[:, q_start:q_end, :]
            # 初始化 O_i, l_i, m_i
            O_i = torch.zeros(batch_size, q_end - q_start, d_model, device=Q.device, dtype=torch.float32)
            l_i = torch.zeros(batch_size, q_end - q_start, device=Q.device, dtype=torch.float32)
            m_i = torch.full((batch_size, q_end - q_start), -float('inf'), device=Q.device, dtype=torch.float32)
            # 遍历 Key/Value tile
            for j in range(T_k):
                k_start, k_end = j * B_k, min((j + 1) * B_k, n_keys)
                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]
                S_ij = (Q_i @ K_j.transpose(-2, -1)) * scale
                # 更新 m_i, l_i (Online Softmax)
                m_i_new = torch.max(m_i, S_ij.max(dim=-1).values)
                P_tilde_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))
                exp_diff = torch.exp(m_i - m_i_new)
                l_i_new = exp_diff.unsqueeze(-1) * l_i.unsqueeze(-1) + P_tilde_ij.sum(dim=-1, keepdim=True)
                # 更新 O_i
                O_i = torch.diag_embed(exp_diff) @ O_i + (P_tilde_ij @ V_j)
                # 更新l_i, m_i
                l_i = l_i_new.squeeze(-1)
                m_i = m_i_new
            # 循环结束后，最终的 O_i 要用 l_i 归一化
            O_i_norm = O_i / l_i.unsqueeze(-1)
            # L_i logsumexp
            L_i = m_i + torch.log(l_i)
            # 将计算好的 tile 写回最终输出
            O[:, q_start:q_end, :] = O_i_norm.to(Q.dtype)
            L[:, q_start:q_end] = L_i
        # 为反向传播保存需要的张量
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        d_model = Q.shape[-1]
        scale = d_model ** -0.5
        
        # 计算 D
        D = torch.sum(dO * O, dim=-1, keepdim=True)
        # 重计算 S
        S = (Q @ K.transpose(-2, -1)) * scale
        if is_causal:
            batch_size, n_queries, n_keys = S.shape
            mask = torch.tril(torch.ones(n_queries, n_keys, device=Q.device)).bool()
            S = S.masked_fill(~mask, -float('inf'))
        # 重计算 P
        P = torch.exp(S - L.unsqueeze(-1))
        # 计算梯度
        dV = P.transpose(-2, -1) @ dO
        dP = dO @ V.transpose(-2, -1)
        dS = P * (dP - D)
        dQ = (dS @ K) * scale
        dK = (dS.transpose(-2, -1) @ Q) * scale
        
        return dQ, dK, dV, None


import triton
import triton.language as tl

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_queries, d_model = Q.shape
        _, n_keys, _ = K.shape
        # 定义 tile size， 必须是编译时的常量
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        T_q = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, n_queries, device=Q.device, dtype=torch.float32)
        # define Launch Grid
        grid = (T_q, batch_size)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale=d_model**-0.5,
            D=d_model,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        ctx.save_for_backward(Q, K, V, O ,L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        # 1. 从 ctx 恢复保存的量
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        d_model = Q.shape[-1]
        scale = d_model ** -0.5
        
        # 用 pytorch compile
        @torch.compile
        def _backward(Q, K, V, O, L, dO, is_causal):
            # 2. 计算 D=rowsum(dO*O)
            # D (batch_size, n_queries, 1)
            D = torch.sum(dO * O, dim=-1, keepdim=True)
            # recompute S P
            # S = QK^T/sqrt(d)
            S = (Q @ K.transpose(-2, -1)) * scale
            if is_causal:
                batch_size, n_queries, n_keys = S.shape
                mask = torch.tril(torch.ones(n_queries, n_keys, device=Q.device)).bool()
                S = S.masked_fill(~mask, -float('inf'))
            # P = exp(S - L)
            P = torch.exp(S - L.unsqueeze(-1))
            # 4. 计算梯度
            # dV = P^T @ dO
            dV = P.transpose(-2, -1) @ dO
            # dP = dO @ V^T
            dP = dO @ V.transpose(-2, -1)
            #dS = P * (dP - D)
            dS = P * (dP - D)
            # dQ = dS @ K * sqrt(d)
            dQ = (dS @ K) * scale
            # dK = dS^T @ Q / sqrt(d)
            dK = (dS.transpose(-2, -1) @ Q) * scale
            return dQ, dK, dV
    
        dQ, dK, dV = _backward(Q, K, V, O, L, dO, is_causal)
        return dQ, dK, dV, None
    
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # 程序索引
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    T_k = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE
    # 初始化 ptr
    q_start = query_tile_index * Q_TILE_SIZE
    q_offsets = q_start + tl.arange(0, Q_TILE_SIZE)
    d_offsets = tl.arange(0, D)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_ptrs = L_ptr + batch_index * stride_lb + q_offsets
    # 初始化累加器
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], -float('inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1))
    # 主循环
    for j in range(T_k):
        k_start = j * K_TILE_SIZE
        K_j = tl.load(K_block_ptr, boundary_check=(1,0))
        V_j = tl.load(V_block_ptr, boundary_check=(0,1))
        S_ij = tl.dot(Q_i, K_j) * scale
        if is_causal:
            q_indices = q_start + tl.arange(0, Q_TILE_SIZE)
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] < k_indices[None, :]
            S_ij = tl.where(causal_mask, -float('inf'), S_ij)

        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        P_tilde_ij = tl.exp(S_ij - m_i_new[:, None])
        exp_diff = tl.exp(m_i - m_i_new)
        l_i_new = exp_diff * l_i + tl.sum(P_tilde_ij, 1)
        P_tilde_ij = P_tilde_ij.to(V_j.dtype)
        acc = acc * exp_diff[:, None]
        acc += tl.dot(P_tilde_ij, V_j)
        l_i = l_i_new
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    acc = acc / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, acc.to(O_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(L_ptrs, L_i, mask=q_offsets < N_QUERIES)