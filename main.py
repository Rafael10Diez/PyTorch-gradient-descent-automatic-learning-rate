
# ------------------------------ Tic-Toc profiler ------------------------------

from time import time

class TicToc:
   def __init__(self):
       self.A      = []
       self.output = []
       self.t_ref  = time()
   def tic(self, label=None):
       t              = time()-self.t_ref
       self.A.append([t,label])
       self.output.append(['tic:', label, t])
   def toc(self, label=None):
       t              = time()-self.t_ref
       t0,label_check = self.A.pop()
       assert label == label_check
       self.output.append(['toc:', label, t-t0])
   def print(self):
       for row in self.output: print(*row)
  
# ------------------------------ Optimizer ------------------------------

import torch 

def dynamic_GD_optimizer(cost_function, x, iters_max = 10_000, lr_ini = 1e-8, lr_stop=1e-12, lr_gain=1.5, lr_loss=0.5):
   
   cost_function(x).backward()
   
   Jcost_old = float('inf')
   lr    = lr_ini + 0. # will grow exponentially
   iters = 0
   
   while (lr > lr_stop):
       if iters > iters_max:
           print('WARNING: iteration limit exceeded')
           break
       iters   +=  1
       x.grad  *=  0.
       cost_function(x).backward()
       x.data  -= lr * x.grad
   
       Jcost_now = float(cost_function(x))
   
       if Jcost_now < Jcost_old:
           lr        *=  lr_gain
           x_old      =  x.data + 0. # triggers copy
           Jcost_old  =  Jcost_now
       else:
           lr      *=  lr_loss
           x.data  *=  0.
           x.data  +=  x_old
   return x

# ------------------------------ Test Case (Rosenbrock function) ------------------------------

if __name__ == '__main__':
   def cost_function(xy, a=1, b=100):
       x,y = xy
       return (a-x)**2 + b*(y-x**2)**2
   x = torch.tensor([2.,2],
                    dtype=torch.double,
                    device='cpu', 
                    requires_grad=True)
   timer = TicToc()
   timer.tic('dynamic_GD_optimizer')
   dynamic_GD_optimizer(cost_function, x, iters_max = 100_000)
   timer.toc('dynamic_GD_optimizer')
   timer.print()

   print('\nOptimization residual error (analytical): ', float((x-1).abs().max()))
