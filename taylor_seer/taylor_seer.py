import torch
import functools
import math

def taylor_seer(warmup_steps=1, skip_interval_steps=1, compute_step_map=None, n_derivatives = 2):
    """
    A decorator that approximates the forward pass of an nn.Module to reduce computation.
    
    Args:
        warmup: Number of steps to compute the actual forward pass before starting approximation
        skip_interval: After warmup, compute the actual forward pass every 'skip_interval' steps
        compute_step_map: A list of booleans that indicates whether to compute the actual forward pass for each step
                          if compute_step_map is provided, warmup_steps and skip_interval_steps
        n_derivatives: The number of derivatives to approximate


        for example, if warmup_steps=3 and skip_interval_steps=2, then the equivalent compute_step_map is:
        [
            True,  # compute step 0
            True,  # compute step 1
            True,  # compute step 2
            False, # approximate step 3
            True,  # compute step 4
            False, # approximate step 5
            True,  # compute step 6
            False, # approximate step 7
            True,  # compute step 8
            ...
        ]
        
    Returns:
        A decorator function that can be applied to an nn.Module
    """
    
    # make sure the warmup and skip interval are at least 1
    warmup_steps = max(int(warmup_steps), 1)
    skip_interval_steps = max(int(skip_interval_steps), 1)
    
    # 'order' of the taylor approximation is the value itself, plus however many
    # derivatives we are approximating
    ORDER = n_derivatives + 1

    def decorator(cls):
        original_init = cls.__init__
        original_forward = cls.forward
        
        @functools.wraps(cls.__init__)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.reset_cache()
    
        def reset_cache(self):
            """Reset the cache state between predictions."""
            self.state = {
                'dY_prev': [None] * ORDER,
                'dY_current': [None] * ORDER,
            }
            self.current_step = -1
            self.last_non_approximated_step = -1
    
        def _should_compute_full(step):
            if compute_step_map is not None:
                return compute_step_map[step]
            else:
                # we compute the actual forward pass for the first warmup steps
                # and then we compute the forward pass every skip_interval after that
                if (step < warmup_steps or 
                    (step >= warmup_steps and (step - warmup_steps + 1) % skip_interval_steps == 0)):
                    return True
                else:
                    return False
        
        def _approximate_derivative(Y, dY_prev, current_step, last_non_approximated_step):
            """
            Approximate the derivative of Y using the previous derivatives
            
            Args:
                Y: current value of the feature, i.e. Y=f(X) where f could be a transformer or linear layer
                dY_prev: the value of the derivative of Y t steps ago
                elapsed_steps: number of steps between Y and dY_prev
            """
            dY_current = [None] * ORDER
            dY_current[0] = Y
            
            finite_difference_window = current_step - last_non_approximated_step
            
            for i in range(n_derivatives):
                if dY_prev[i] is not None and current_step > 1:
                    # equation (7) from the paper
                    dY_current[i+1] = (dY_current[i] - dY_prev[i]) / finite_difference_window
                else:
                    break
            return dY_current

        def _approximate_value(dY_current, elapsed_steps):
            """
            Approximate the current value of Y using our current estimate of the derivative
            and the # of timesteps that have passed since the derivative was computed
            
            Args:
                dY_current: the value of the derivatives of Y
                elapsed_steps: number of steps between Y and dY_prev
            """
            # taylor series formula
            output = 0
            for i in range(len(dY_current)):
                if dY_current[i] is not None:
                    output += (1 / math.factorial(i)) * dY_current[i] * (elapsed_steps ** i)
                else:
                    break
            return output


        @functools.wraps(cls.forward)
        def new_forward(self, *args, **kwargs):
            self.current_step += 1
            
            if _should_compute_full(self.current_step):

                # compute actual forward pass
                Y = original_forward(self, *args, **kwargs)
                assert isinstance(Y, torch.Tensor), 'output of decorated forward method must be a torch.Tensor'
                
                self.state['dY_prev'] = self.state['dY_current']
                
                # calculate and update derivative based on present model output and previous derivatives
                self.state['dY_current'] = _approximate_derivative(Y, self.state['dY_prev'], self.current_step, self.last_non_approximated_step)
                
                # reset the finite difference window
                self.last_non_approximated_step = self.current_step
                return Y
                
            # approximate the value of the forward pass using the derivative computed 'finite_difference_window' steps ago
            else:
                finite_difference_window = self.current_step - self.last_non_approximated_step
                assert self.state['dY_current'][0] is not None
                return _approximate_value(self.state['dY_current'], finite_difference_window)
                
        # Replace methods
        cls.__init__ = new_init
        cls.forward = new_forward
        cls.reset_cache = reset_cache
        return cls
    
    return decorator


