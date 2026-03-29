# bpnet.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains a reference implementation of BPNet that can be used
or adapted for your own circumstances. The implementation takes in a
stranded control track and makes predictions for stranded outputs.
"""

import h5py
import time 
import numpy
import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss
from .losses import _mixture_loss

from .performance import pearson_corr
from .performance import calculate_performance_measures
from .logging import Logger
from .bpnet import BPNet

from tqdm import tqdm

from tangermeme.predict import predict

torch.backends.cudnn.benchmark = True


class ControlWrapper(torch.nn.Module):
	"""This wrapper automatically creates a control track of all zeroes.

	This wrapper will check to see whether the model is expecting a control
	track (e.g., most BPNet-style models) and will create one with the expected
	shape. If no control track is expected then it will provide the normal
	output from the model.
	"""

	def __init__(self, model):
		super(ControlWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None):
		if X_ctl != None:
			return self.model(X, X_ctl)

		if self.model.n_control_tracks == 0:
			return self.model(X)

		X_ctl = torch.zeros(X.shape[0], self.model.n_control_tracks,
			X.shape[-1], dtype=X.dtype, device=X.device)
		return self.model(X, X_ctl)

	

class _ProfileLogitScaling(torch.nn.Module):
	"""This ugly class is necessary because of Captum.

	Captum internally registers classes as linear or non-linear. Because the
	profile wrapper performs some non-linear operations, those operations must
	be registered as such. However, the inputs to the wrapper are not the
	logits that are being modified in a non-linear manner but rather the
	original sequence that is subsequently run through the model. Hence, this
	object will contain all of the operations performed on the logits and
	can be registered.


	Parameters
	----------
	logits: torch.Tensor, shape=(-1, -1)
		The logits as they come out of a Chrom/BPNet model.
	"""

	def __init__(self):
		super(_ProfileLogitScaling, self).__init__()
		self.softmax = torch.nn.Softmax(dim=-1)

	def forward(self, logits):
		y_softmax = self.softmax(logits)
		return logits * y_softmax


class ProfileWrapper(torch.nn.Module):
	"""A wrapper class that returns transformed profiles.

	This class takes in a trained model and returns the weighted softmaxed
	outputs of the first dimension. Specifically, it takes the predicted
	"logits" and takes the dot product between them and the softmaxed versions
	of those logits. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(ProfileWrapper, self).__init__()
		self.model = model
		self.flatten = torch.nn.Flatten()
		self.scaling = _ProfileLogitScaling()

	def forward(self, X, X_ctl=None, **kwargs):
		logits = self.model(X, X_ctl, **kwargs)[0]
		logits = self.flatten(logits)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)
		return self.scaling(logits).sum(dim=-1, keepdims=True)


class CountWrapper(torch.nn.Module):
	"""A wrapper class that only returns the predicted counts.

	This class takes in a trained model and returns only the second output.
	For BPNet models, this means that it is only returning the count
	predictions. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(CountWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None, **kwargs):
		return self.model(X, X_ctl, **kwargs)[1]


class BiasFactorizedBPNet(BPNet):
	"""A basic BPNet model with stranded profile and total count prediction.

	This is a reference implementation for BPNet models. It exactly matches the
	architecture in the official ChromBPNet repository. It is very similar to
	the implementation in the official basepairmodels repository but differs in
	when the activation function is applied for the resifual layers. See the
	BasePairNet object below for an implementation that matches that repository. 

	The model takes in one-hot encoded sequence, runs it through: 

	(1) a single wide convolution operation 

	THEN 

	(2) a user-defined number of dilated residual convolutions

	THEN

	(3a) profile predictions done using a very wide convolution layer 
	that also takes in stranded control tracks 

	AND

	(3b) total count prediction done using an average pooling on the output
	from 2 followed by concatenation with the log1p of the sum of the
	stranded control tracks and then run through a dense layer.

	This implementation differs from the original BPNet implementation in
	two ways:

	(1) The model concatenates stranded control tracks for profile
	prediction as opposed to adding the two strands together and also then
	smoothing that track 

	(2) The control input for the count prediction task is the log1p of
	the strand-wise sum of the control tracks, as opposed to the raw
	counts themselves.

	(3) A single log softmax is applied across both strands such that
	the logsumexp of both strands together is 0. Put another way, the
	two strands are concatenated together, a log softmax is applied,
	and the MNLL loss is calculated on the concatenation. 

	(4) The count prediction task is predicting the total counts across
	both strands. The counts are then distributed across strands according
	to the single log softmax from 3.


	Parameters
	----------
	n_filters: int, optional
		The number of filters to use per convolution. Default is 64.

	n_layers: int, optional
		The number of dilated residual layers to include in the model.
		Default is 8.

	n_outputs: int, optional
		The number of profile outputs from the model. Generally either 1 or 2 
		depending on if the data is unstranded or stranded. Default is 2.

	n_control_tracks: int, optional
		The number of control tracks to feed into the model. When predicting
		TFs, this is usually 2. When predicting accessibility, this is usualy
		0. When 0, this input is removed from the model. Default is 2.

	count_loss_weight: float, optional
		The weight to put on the count loss.

	profile_output_bias: bool, optional
		Whether to include a bias term in the final profile convolution.
		Removing this term can help with attribution stability and will usually
		not affect performance. Default is True.

	count_output_bias: bool, optional
		Whether to include a bias term in the linear layer used to predict
		counts. Removing this term can help with attribution stability but
		may affect performance. Default is True.

	name: str or None, optional
		The name to save the model to during training.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.

	verbose: bool, optional
		Whether to display statistics during training. Setting this to False
		will still save the file at the end, but does not print anything to
		screen during training. Default is True.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
		n_control_tracks=2, count_loss_weight=1, profile_output_bias=True,
		count_output_bias=True, name=None, trimming=None, verbose=True,
		frozen_bias_model=None):
		if frozen_bias_model is None:
			raise ValueError("frozen_bias_model must be provided for BiasFactorizedBPNet.")
		super(BiasFactorizedBPNet, self).__init__(
			n_filters=n_filters,
			n_layers=n_layers,
			n_outputs=n_outputs,
			n_control_tracks=n_control_tracks,
			count_loss_weight=count_loss_weight,
			profile_output_bias=profile_output_bias,
			count_output_bias=count_output_bias,
			name=name,
			trimming=trimming,
			verbose=verbose,
		)
		self.frozen_bias_model = frozen_bias_model
		self._freeze_bias_model()


	def _freeze_bias_model(self):
		self.frozen_bias_model.eval()
		for p in self.frozen_bias_model.parameters():
			p.requires_grad = False


	def train(self, mode=True):
		super(BiasFactorizedBPNet, self).train(mode)
		self._freeze_bias_model()
		return self


	def forward(self, X, X_ctl=None):
		"""A forward pass of the model.

		This method takes in a nucleotide sequence X, a corresponding
		per-position value from a control track, and a per-locus value
		from the control track and makes predictions for the profile 
		and for the counts. This per-locus value is usually the
		log(sum(X_ctl_profile)+1) when the control is an experimental
		read track but can also be the output from another model.

		Parameters
		----------
		X: torch.tensor, shape=(batch_size, 4, length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
			A value representing the signal of the control at each position in 
			the sequence. If no controls, pass in None. Default is None.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
			The output predictions for each strand trimmed to the output
			length.
		"""

		X_input = X
		y_profile_residual, y_counts = super(BiasFactorizedBPNet, self).forward(X, X_ctl)

		with torch.no_grad():
			bias_out = self.frozen_bias_model(X_input)
			if isinstance(bias_out, (tuple, list)):
				y_profile_bias = bias_out[0]
			else:
				y_profile_bias = bias_out

		if y_profile_bias.shape[1] != y_profile_residual.shape[1]:
			raise ValueError(
				f"Channel mismatch: bias={y_profile_bias.shape[1]}, "
				f"residual={y_profile_residual.shape[1]}"
			)
		if y_profile_bias.shape[-1] != y_profile_residual.shape[-1]:
			target_len = min(y_profile_bias.shape[-1], y_profile_residual.shape[-1])
			bias_start = (y_profile_bias.shape[-1] - target_len) // 2
			res_start = (y_profile_residual.shape[-1] - target_len) // 2
			y_profile_bias = y_profile_bias[:, :, bias_start:bias_start+target_len]
			y_profile_residual = y_profile_residual[:, :, res_start:res_start+target_len]

		y_profile = y_profile_bias + y_profile_residual
		return y_profile, y_counts


	def fit(self, training_data, optimizer, scheduler=None, X_valid=None, 
		X_ctl_valid=None, y_valid=None, max_epochs=100, batch_size=64, 
		dtype='float32', device='cuda', early_stopping=None):
		return super(BiasFactorizedBPNet, self).fit(
			training_data=training_data,
			optimizer=optimizer,
			scheduler=scheduler,
			X_valid=X_valid,
			X_ctl_valid=X_ctl_valid,
			y_valid=y_valid,
			max_epochs=max_epochs,
			batch_size=batch_size,
			dtype=dtype,
			device=device,
			early_stopping=early_stopping,
		)
