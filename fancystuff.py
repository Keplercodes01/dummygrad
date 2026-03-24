import dummygrad as dummy

class Tensor(dummy.Tensor):
    def show(self):
        def recursive_print(data, shape, offset = 0, indent = 0):
            if len(shape) == 1:
                row = [f"{data[offset + i]:.4f}" for i in range(shape[0])]
                print(" " * indent + "[" + ", ".join(row) + "]", end = " ")
            else: 
                print(" " * indent + "[")
                stride = 1
                for s in shape[1:]:
                    stride *= s
                for i in range(shape[0]):
                    recursive_print(data, shape[1:], offset + i * stride, indent + 2)
                    if i < shape[0]-1:
                        print(",")
                    else:
                        print()
                print(" " * indent + "]", end = "") 

        recursive_print(self.data, self.shape)
        print(f"\nshape={self.shape}")

    def show_grad(self):
        def recursive_print(data, shape, offset = 0, indent = 0):
            if len(shape) == 1:
                row = [f"{data[offset + i]:.4f}" for i in range(shape[0])]
                print(" " * indent + "[" + ", ".join(row) + "]", end = " ")
            else: 
                print(" " * indent + "[")
                stride = 1
                for s in shape[1:]:
                    stride *= s
                for i in range(shape[0]):
                    recursive_print(data, shape[1:], offset + i * stride, indent + 2)
                    if i < shape[0]-1:
                        print(",")
                    else:
                        print()
                print(" " * indent + "]", end = "") 

        recursive_print(self.grad, self.shape)
        print(f"\nshape={self.shape}")

