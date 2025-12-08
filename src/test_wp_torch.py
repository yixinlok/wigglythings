import warp as wp
import torch

num = 2

@wp.kernel()
def loss(xs: wp.array(dtype=float, ndim=num), l: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(l, 0, xs[tid, 0] ** 2.0 + xs[tid, 1] ** 2.0)

# indicate requires_grad so that Warp can accumulate gradients in the grad buffers
xs = torch.randn(100, num)
l = torch.zeros(1)
opt = torch.optim.Adam([xs], lr=0.1)

# wp_xs = wp.from_torch(xs)
# wp_l = wp.from_torch(l)

# tape = wp.Tape()
# with tape:
    # record the loss function kernel launch on the tape
wp.launch(loss, dim=len(xs), inputs=[xs], outputs=[l], device="cpu")

print(f"loss: {l.item()}")
# for i in range(1):
#     tape.zero()
#     tape.backward(loss=wp_l)  # compute gradients
#     # now xs.grad will be populated with the gradients computed by Warp
#     opt.step()  # update xs (and thereby wp_xs)

#     # these lines are only needed for evaluating the loss
#     # (the optimization just needs the gradient, not the loss value)
#     wp_l.zero_()
#     wp.launch(loss, dim=len(xs), inputs=[wp_xs], outputs=[wp_l], device="cpu")
#     print(f"{i}\tloss: {l.item()}")