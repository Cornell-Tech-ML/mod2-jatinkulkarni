"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        self.history = History()

    def requires_grad(self) -> bool:
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        # print(f"🙊🙊🙊🙊🙊🙊🙊 b: {b}")
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        print(f"🟤🟤🟤🟤🟤🟤🟤🟤🟤🟤🟤 h.inputs:{h.inputs} x: {x}")
        result = []
        for inp, d_in in zip(h.inputs, x):
            print(f"🟥🟥🟥🟥🟥🟥🟥🟥🟥 inp:{inp} d_in:{d_in}")
            val2 = inp.expand(self._ensure_tensor(d_in))
            print(f"🟧🟧🟧🟧🟧🟧🟧🟧🟧 val2: {val2}")
            result.append((inp, val2))

        return result
        # return [
        #     (inp, inp.expand(self._ensure_tensor(d_in)))
        #     for inp, d_in in zip(h.inputs, x)
        # ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        print("🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵")
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
            print(f"🟪🟪🟪🟪🟪🟪🟪🟪🟪🟪 Grad output was none: {grad_output}")
        print("🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣🟣")
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # TODO: Implement for Task 2.3.
    """
    ✅ add
    ✅ sub
    ✅ mul
    ✅ lt
    ✅ eq
    ✅ gt
    ✅ neg
    ✅ radd
    ✅ rmul
    ✅ all
    ✅ is_close
    ✅ sigmoid
    ✅ relu
    ✅ log
    ✅ exp

    Should take an optional dim argument:
    ✅ sum
    ✅ mean
    ✅ permute
    ✅ view

    Should set .grad to None - zero_grad_
    """
    @property
    def size(self) -> int:
        """Return the total number of elements in the tensor."""
        return int(np.prod(self.shape))

    @property
    def dims(self) -> int:
        """Return the number of dimensions in the tensor."""
        return len(self.shape)



    def __add__(self, y: Tensor) -> Tensor:
        """Adding self Tensor and y Tensor"""
        y = self._ensure_tensor(y)
        # y.zero_grad_()
        return Add.apply(self, y)

    def __sub__(self, y: TensorLike) -> Tensor:
        """Subtracting self Tensor by y Tensor"""
        y = self._ensure_tensor(y)
        return Add.apply(self, Neg.apply(y))

    def __mul__(self, y: TensorLike) -> Tensor:
        """Multiplying self Tensor and y Tensor"""
        y = self._ensure_tensor(y)
        return Mul.apply(self, y)

    def __lt__(self, y: TensorLike) -> Tensor:
        """Checking Less Than operation on self Tensor and y Tensor"""
        y = self._ensure_tensor(y)
        return LT.apply(self, y)

    def __eq__(self, y: TensorLike) -> Tensor:
        """Checing Equal To operation on self Tensor and y Tensor"""
        y = self._ensure_tensor(y)
        return EQ.apply(self, y)

    def __gt__(self, y: TensorLike) -> Tensor:
        """Checking Greater Than operation on self Tensor and y Tensor"""
        y = self._ensure_tensor(y)
        return LT.apply(y, self)

    def __neg__(self) -> Tensor:
        """Negate self Tensor"""
        return Neg.apply(self)

    def __radd__(self, y: Tensor) -> Tensor:
        """Right adding self Tensor and y Tensor"""
        return self.__add__(y)

    def __rmul__(self, y: Tensor) -> Tensor:
        """Right multiplying self Tensor and y Tensor"""
        return self.__mul__(y)

    def all(self, dim: Tensor = None) -> Tensor:
        """Return All for self Tensor"""
        if dim is not None:
            dim = self._ensure_tensor(dim)
            return All.apply(self, dim)
        return All.apply(self)

    def is_close(self, y: Tensor) -> Tensor:
        """Checks if both tensors are close"""
        y = self._ensure_tensor(y)
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies Sigmoid to self Tensor"""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies ReLU to self Tensor"""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Appllies Log to self Tensor"""
        return Log.apply(self)
    
    def exp(self) -> Tensor:
        """Applies Exp to self Tensor"""
        return Exp.apply(self)

    def sum(self, dim: Tensor = None) -> Tensor:
        """Applies sum to self, tensor"""
        if dim is not None:
            dim = self._ensure_tensor(dim)
            return Sum.apply(self, dim)
        return Sum.apply(self)

    def mean(self, dim: Tensor = None) -> Tensor:
        """Calculates mean of Tensor"""
        summed = self.sum(dim)
        if dim is None:
            num_elements = self.size 
        else:
            num_elements = self.shape[dim]
        return summed / num_elements

    def permute(self, *dims: int) -> Tensor:
        """Permute Tensor"""
        # if len(dims) != len(self.shape):
        #     raise ValueError(f"Expected {len(self.shape)} dimensions but got {len(dims)}")
        
        # Check if the tensor is 1-dimensional and doesn't require permutation
        if len(self.shape) == 1:
            return self  # No permutation is needed for a single-element tensor

        dims_tensor = Tensor.make(list(dims), (len(dims),), backend=self.backend)
        return Permute.apply(self, dims_tensor)



    def view(self, dim: Tensor = None) -> Tensor:
        """Performs view on Tensor"""
        if dim is not None:
            dim = self._ensure_tensor(dim)
            return View.apply(self, dim)
        return View.apply(self)

    def zero_grad_(self) -> None:
        """Sets grad to None"""
        self.grad = None


    