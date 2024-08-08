import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import numpy as np
from collections import defaultdict

from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers, dice


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""


    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )
        
        self.use_4d_image = kwargs["4d_input"] if "4d_input" in kwargs else False
        self.sdf_alpha = kwargs["sdf_alpha"] if "sdf_alpha" in kwargs else self.args["sdf_alpha"]

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if False: #self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )
        self.dice_loss = dice.MultiClassDiceLoss(labels=[1,2,3,4,5,7,8])
        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.jacobian_symmetric = (
            kwargs["jacobian_symmetric"]
            if "jacobian_symmetric" in kwargs
            else self.args["jacobian_symmetric"]
        )
        
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image
        shape = self.fixed_image.shape if not self.use_4d_image else self.fixed_image.shape[:-1]
        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
            self.mask, shape
        )

        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()
    
    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # Variables specific to this class
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["batch_size"] = 10000
        self.args["layers"] = [3, 256, 256, 256, 3]
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["jacobian_regularization"] = False
        self.args["jacobian_symmetric"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "MLP"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1
        
        self.args["4d_input"] = False
        self.args["sdf_alpha"] = 0.0

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
        )[: self.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)

        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp
        
        if self.use_4d_image:
            transformed_image = general.fast_trilinear_interpolation_4D(
                self.moving_image,
                coord_temp[:, 0],
                coord_temp[:, 1],
                coord_temp[:, 2],
            )
            fixed_image = general.fast_trilinear_interpolation_4D(
                self.fixed_image,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )
        else:
            transformed_image = self.transform_no_add(coord_temp)
            fixed_image = general.fast_trilinear_interpolation(
                self.fixed_image,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )

        # Compute the loss
        if not self.use_4d_image:
            loss += self.criterion(transformed_image, fixed_image)
        else:
            hu_loss = (1-self.sdf_alpha) * self.criterion(transformed_image[...,0], fixed_image[...,0])
            sdf_loss =   self.sdf_alpha  * self.criterion(transformed_image[...,1], fixed_image[...,1]) 
            
            # sdf_loss = self.sdf_alpha * self.dice_loss(general.fast_nearest_neighbor_interpolation(self.moving_image[...,1], coord_temp[:, 0], coord_temp[:, 1], coord_temp[:, 2]),
            #                                            general.fast_nearest_neighbor_interpolation(self.fixed_image[...,1], coordinate_tensor[:, 0], coordinate_tensor[:, 1], coordinate_tensor[:, 2]))
            loss += hu_loss + sdf_loss
            self.losses["hu_loss"].append(hu_loss.detach().cpu().numpy())
            self.losses["sdf_loss"].append(sdf_loss.detach().cpu().numpy())
            self.losses["total_loss"].append(loss.detach().cpu().numpy())
            
        # loss += self.criterion(transformed_image, fixed_image) if not self.use_4d_image \
        #     else (1-self.sdf_alpha) * self.criterion(transformed_image[...,0], fixed_image[...,0]) \
        #         +   self.sdf_alpha  * self.criterion(transformed_image[...,1], fixed_image[...,1])
        
        # TODO: add sdf loss
        
        # TODO: or dice loss  

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)

        # Regularization
        if self.jacobian_regularization:
            jac_loss = self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss += jac_loss
            self.losses["jacobian_loss"].append(jac_loss.detach().cpu().numpy())
        if self.jacobian_symmetric:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_symmetric_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            loss += self.alpha_bending * regularizers.compute_bending_energy(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )

        # Perform the backpropagation and update the parameters accordingly

        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform(
        self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        self.losses = defaultdict(list)
        pbar = tqdm.tqdm(range(epochs), ncols=100)
        for i in pbar:
            self.training_iteration(i)
            pbar.set_postfix(loss=f"{self.loss_list[i]:.6}")

    def transform_volume(self, dims, image=None):
    
        coordinate_tensor = general.make_coordinate_tensor(dims)
        coordinate_chunks = torch.split(coordinate_tensor, self.batch_size)
        outputs = []
        for chunk in coordinate_chunks:
            output = self.network(chunk)
            # coord_temp = torch.add(output, chunk)
            outputs.append(output.cpu().detach())
            

        outputs = torch.cat(outputs).cuda()

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(outputs, coordinate_tensor)
        
        if image is not None:
            
            im = torch.tensor(image.astype(float), device="cuda").float() if isinstance(image, np.ndarray) else image.cuda()

            transformed_image = general.fast_nearest_neighbor_interpolation(im, coord_temp[:, 0], coord_temp[:, 1], coord_temp[:, 2]) if np.issubdtype(np.array(image).dtype, np.integer) \
                            else general.fast_trilinear_interpolation(im, coord_temp[:, 0], coord_temp[:, 1], coord_temp[:, 2])
        else:
            transformed_image = self.transform_no_add(coord_temp)
            
        return (
            transformed_image
            .cpu()
            .detach()
            .numpy()
            .reshape(dims)
        )
        
    def transform_points(self, points):
        
        coordinate_tensor = torch.tensor(points, device="cuda").float()
        coordinate_chunks = torch.split(coordinate_tensor, self.batch_size)
        outputs = []
        for chunk in coordinate_chunks:
            output = self.network(chunk)
            # coord_temp = torch.add(output, chunk)
            outputs.append(output.cpu().detach())
            

        outputs = torch.cat(outputs).cuda()

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(outputs, coordinate_tensor)
        return coord_temp.cpu().detach().numpy()
    
    def caLculate_negative_jacobian(self):
        """Calculate the negative jacobian of the network."""
        
        
        # Create coordinate tensor
        coordinate_tensor = self.possible_coordinate_tensor
        coordinate_tensor.requires_grad_(True)
        coordinate_chunks = torch.split(coordinate_tensor, self.batch_size)
        outputs = []
        jacobians = []
        for chunk in coordinate_chunks:
            output = self.network(chunk)
            jacobian_matrix = regularizers.compute_jacobian_matrix(chunk, output)
            
            jacobians.append(jacobian_matrix.cpu().detach())
            
        jacobians = torch.cat(jacobians)
        jacobian = torch.det(jacobians)
        return (jacobian < 0).sum(), jacobian.shape[0]
    
    def save_network(self, filename="network.pth"):
        """Save the network to a file."""

        torch.save({"state_dict": self.network.state_dict(),
                    "losses": self.losses,
                    },
                    filename)
        
    def load_network(self, filename="network.pth"):
        """Load the network from a file."""
        
        checkpoint = torch.load(filename)
        if "state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["state_dict"])
            self.losses = checkpoint["lossses"] if "lossses" in checkpoint else checkpoint["losses"]
        else:
            self.network.load_state_dict(checkpoint)
            
        if self.gpu:
            self.network.cuda()