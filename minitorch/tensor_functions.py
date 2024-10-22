"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for Negation method"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward method for Negation Method"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for Inverse Method"""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward method for Inverse Method"""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for Add method"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward method for Add Method"""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for Tensor Multiplication"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward method for Tensor Multiplication"""
        #  raise NotImplementedError("Need to implement for Task 2.3")
        t1, t2 = ctx.saved_tensors

        grad_t1 = grad_output.f.mul_zip(grad_output, t2)
        grad_t2 = grad_output.f.mul_zip(grad_output, t1)

        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Forward method for Tensor Sigmoid"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t)
        return t.f.sigmoid_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward method for Tensor Sigmoid"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        (t,) = ctx.saved_tensors
        grad_t = t.f.mul_zip(
            t.f.sigmoid_map(t), (t._ensure_tensor(1) - t.f.sigmoid_map(t))
        )
        return grad_t * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Forward method for Tensor ReLU"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t)
        return t.f.relu_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward method for Tensor ReLU"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        (t,) = ctx.saved_tensors
        return t.f.relu_back_zip(t, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Forward method for Tensor Log"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t)
        return t.f.log_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward Method for Tensor Log"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        (t,) = ctx.saved_tensors
        return t.f.log_back_zip(t, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Forward method for Tensor Exp"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t)
        return t.f.exp_map(t)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward method for Tensor Exp"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        (t,) = ctx.saved_tensors
        return t.f.mul_zip(t.f.exp_map(t), grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, dim: Tensor = None) -> Tensor:
        """Forward method for Tensor Sum"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t, dim)
        if dim is not None:
            dim = dim.item()
            return t.f.add_reduce(t, int(dim))
        else:
            return t.f.add_reduce(t.contiguous().view(int(operators.prod(t.shape))), 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor,]:
        """Backward method for Tensor Sum"""
        (t, dim) = ctx.saved_tensors

        if dim is not None:
            dim_item = dim.item()
            shape = list(t.shape)
            shape[int(dim_item)] = 1

            reshaped_grad = grad_output.view(*shape)

            broadcasted_grad = t.expand(reshaped_grad)

            return broadcasted_grad, dim

        else:
            broadcasted_grad = t.expand(grad_output)
            return broadcasted_grad


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for Tensor LT"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward method for Tensor LT"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        (t1, t2) = ctx.saved_tensors
        zero_t1 = zeros(t1.shape, t1.backend)
        zero_t2 = zeros(t2.shape, t2.backend)
        return zero_t1, zero_t2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for Tensor EQ"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward method for Tensor EQ"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        (t1, t2) = ctx.saved_tensors
        zero_t1 = zeros(t1.shape, t1.backend)
        zero_t2 = zeros(t2.shape, t1.backend)
        return zero_t1, zero_t2


class IsClose(Function):
    @staticmethod
    def forward(
        ctx: Context, t1: Tensor, t2: Tensor, tolerance: float = 1e-5
    ) -> Tensor:
        """Forward method for Tensor is_close"""
        # raise NotImplementedError("Need to implement for Task 2.3")
        ctx.save_for_backward(t1, t2)
        return t1.f.is_close_zip(t1, t2)

    # No backward function for IsClose
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """There is no Backward method for Tensor is_close"""
        raise NotImplementedError("No backward for IsClose")


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, dims: Optional[Tensor] = None) -> Tensor:
        """Forward method for Tensor Permute"""
        ctx.save_for_backward(dims, t)

        if dims is None:
            ctx.save_for_backward(tuple(range(len(t.shape))))
            return t

        dims_numpy = dims.to_numpy()
        dims = tuple(int(dim) for dim in dims_numpy)

        permuted_tensor_data = t._tensor.permute(*dims)

        return minitorch.Tensor.make(
            permuted_tensor_data._storage, permuted_tensor_data.shape, backend=t.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward method for Tensor Permute"""
        (dims, t) = ctx.saved_tensors

        if dims is None:
            return grad_output, minitorch.Tensor.make(
                list(range(len(grad_output.shape))),
                grad_output.shape,
                backend=t.backend,
            )

        dims_numpy = dims.to_numpy()
        dims_tuple = tuple(int(dim) for dim in dims_numpy)

        inverse_dims = [dims_tuple.index(i) for i in range(len(dims_tuple))]

        permuted_grad_data = grad_output._tensor.permute(*inverse_dims)

        return minitorch.Tensor.make(
            permuted_grad_data._storage,
            permuted_grad_data.shape,
            backend=grad_output.backend,
        ), dims


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Forward function for View Method"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Gradient check for Tensors"""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
