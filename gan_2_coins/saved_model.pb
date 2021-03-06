??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
}
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???* 
shared_namedense_40/kernel
v
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*!
_output_shapes
:???*
dtype0
t
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_namedense_40/bias
m
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes

:??*
dtype0
?
batch_normalization_99/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_99/gamma
?
0batch_normalization_99/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_99/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_99/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_99/beta
?
/batch_normalization_99/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_99/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_99/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_99/moving_mean
?
6batch_normalization_99/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_99/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_99/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_99/moving_variance
?
:batch_normalization_99/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_99/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameconv2d_transpose_80/kernel
?
.conv2d_transpose_80/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_80/kernel*'
_output_shapes
:@?*
dtype0
?
conv2d_transpose_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_80/bias
?
,conv2d_transpose_80/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_80/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_100/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_100/gamma
?
1batch_normalization_100/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_100/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_100/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_100/beta
?
0batch_normalization_100/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_100/beta*
_output_shapes
:@*
dtype0
?
#batch_normalization_100/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_100/moving_mean
?
7batch_normalization_100/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_100/moving_mean*
_output_shapes
:@*
dtype0
?
'batch_normalization_100/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_100/moving_variance
?
;batch_normalization_100/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_100/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameconv2d_112/kernel

%conv2d_112/kernel/Read/ReadVariableOpReadVariableOpconv2d_112/kernel*&
_output_shapes
:@@*
dtype0
v
conv2d_112/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_112/bias
o
#conv2d_112/bias/Read/ReadVariableOpReadVariableOpconv2d_112/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_101/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_101/gamma
?
1batch_normalization_101/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_101/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_101/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_101/beta
?
0batch_normalization_101/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_101/beta*
_output_shapes
:@*
dtype0
?
#batch_normalization_101/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_101/moving_mean
?
7batch_normalization_101/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_101/moving_mean*
_output_shapes
:@*
dtype0
?
'batch_normalization_101/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_101/moving_variance
?
;batch_normalization_101/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_101/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_81/kernel
?
.conv2d_transpose_81/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_81/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_81/bias
?
,conv2d_transpose_81/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_81/bias*
_output_shapes
: *
dtype0
?
batch_normalization_102/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_102/gamma
?
1batch_normalization_102/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_102/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_102/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_102/beta
?
0batch_normalization_102/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_102/beta*
_output_shapes
: *
dtype0
?
#batch_normalization_102/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_102/moving_mean
?
7batch_normalization_102/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_102/moving_mean*
_output_shapes
: *
dtype0
?
'batch_normalization_102/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_102/moving_variance
?
;batch_normalization_102/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_102/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_82/kernel
?
.conv2d_transpose_82/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_82/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_82/bias
?
,conv2d_transpose_82/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_82/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.regularization_losses
/	variables
0trainable_variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
?
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
?
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
 
?
0
1
2
3
4
5
#6
$7
*8
+9
,10
-11
212
313
914
:15
;16
<17
A18
B19
H20
I21
J22
K23
P24
Q25
?
0
1
2
3
#4
$5
*6
+7
28
39
910
:11
A12
B13
H14
I15
P16
Q17
?
Vlayer_metrics
regularization_losses

Wlayers
	variables
trainable_variables
Xlayer_regularization_losses
Ymetrics
Znon_trainable_variables
 
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
[layer_metrics
regularization_losses
\layer_regularization_losses
	variables
trainable_variables

]layers
^metrics
_non_trainable_variables
 
 
 
?
`layer_metrics
regularization_losses
alayer_regularization_losses
	variables
trainable_variables

blayers
cmetrics
dnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_99/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_99/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_99/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_99/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
?
elayer_metrics
regularization_losses
flayer_regularization_losses
 	variables
!trainable_variables

glayers
hmetrics
inon_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_80/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_80/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
jlayer_metrics
%regularization_losses
klayer_regularization_losses
&	variables
'trainable_variables

llayers
mmetrics
nnon_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_100/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_100/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_100/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_100/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1
,2
-3

*0
+1
?
olayer_metrics
.regularization_losses
player_regularization_losses
/	variables
0trainable_variables

qlayers
rmetrics
snon_trainable_variables
][
VARIABLE_VALUEconv2d_112/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_112/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
?
tlayer_metrics
4regularization_losses
ulayer_regularization_losses
5	variables
6trainable_variables

vlayers
wmetrics
xnon_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_101/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_101/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_101/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_101/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1
;2
<3

90
:1
?
ylayer_metrics
=regularization_losses
zlayer_regularization_losses
>	variables
?trainable_variables

{layers
|metrics
}non_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_81/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_81/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
?
~layer_metrics
Cregularization_losses
layer_regularization_losses
D	variables
Etrainable_variables
?layers
?metrics
?non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_102/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_102/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_102/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_102/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1
J2
K3

H0
I1
?
?layer_metrics
Lregularization_losses
 ?layer_regularization_losses
M	variables
Ntrainable_variables
?layers
?metrics
?non_trainable_variables
fd
VARIABLE_VALUEconv2d_transpose_82/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_82/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
?
?layer_metrics
Rregularization_losses
 ?layer_regularization_losses
S	variables
Ttrainable_variables
?layers
?metrics
?non_trainable_variables
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
8
0
1
,2
-3
;4
<5
J6
K7
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

,0
-1
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
?
serving_default_dense_40_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_40_inputdense_40/kerneldense_40/biasbatch_normalization_99/gammabatch_normalization_99/beta"batch_normalization_99/moving_mean&batch_normalization_99/moving_varianceconv2d_transpose_80/kernelconv2d_transpose_80/biasbatch_normalization_100/gammabatch_normalization_100/beta#batch_normalization_100/moving_mean'batch_normalization_100/moving_varianceconv2d_112/kernelconv2d_112/biasbatch_normalization_101/gammabatch_normalization_101/beta#batch_normalization_101/moving_mean'batch_normalization_101/moving_varianceconv2d_transpose_81/kernelconv2d_transpose_81/biasbatch_normalization_102/gammabatch_normalization_102/beta#batch_normalization_102/moving_mean'batch_normalization_102/moving_varianceconv2d_transpose_82/kernelconv2d_transpose_82/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_19454758
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp0batch_normalization_99/gamma/Read/ReadVariableOp/batch_normalization_99/beta/Read/ReadVariableOp6batch_normalization_99/moving_mean/Read/ReadVariableOp:batch_normalization_99/moving_variance/Read/ReadVariableOp.conv2d_transpose_80/kernel/Read/ReadVariableOp,conv2d_transpose_80/bias/Read/ReadVariableOp1batch_normalization_100/gamma/Read/ReadVariableOp0batch_normalization_100/beta/Read/ReadVariableOp7batch_normalization_100/moving_mean/Read/ReadVariableOp;batch_normalization_100/moving_variance/Read/ReadVariableOp%conv2d_112/kernel/Read/ReadVariableOp#conv2d_112/bias/Read/ReadVariableOp1batch_normalization_101/gamma/Read/ReadVariableOp0batch_normalization_101/beta/Read/ReadVariableOp7batch_normalization_101/moving_mean/Read/ReadVariableOp;batch_normalization_101/moving_variance/Read/ReadVariableOp.conv2d_transpose_81/kernel/Read/ReadVariableOp,conv2d_transpose_81/bias/Read/ReadVariableOp1batch_normalization_102/gamma/Read/ReadVariableOp0batch_normalization_102/beta/Read/ReadVariableOp7batch_normalization_102/moving_mean/Read/ReadVariableOp;batch_normalization_102/moving_variance/Read/ReadVariableOp.conv2d_transpose_82/kernel/Read/ReadVariableOp,conv2d_transpose_82/bias/Read/ReadVariableOpConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_19455627
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_40/kerneldense_40/biasbatch_normalization_99/gammabatch_normalization_99/beta"batch_normalization_99/moving_mean&batch_normalization_99/moving_varianceconv2d_transpose_80/kernelconv2d_transpose_80/biasbatch_normalization_100/gammabatch_normalization_100/beta#batch_normalization_100/moving_mean'batch_normalization_100/moving_varianceconv2d_112/kernelconv2d_112/biasbatch_normalization_101/gammabatch_normalization_101/beta#batch_normalization_101/moving_mean'batch_normalization_101/moving_varianceconv2d_transpose_81/kernelconv2d_transpose_81/biasbatch_normalization_102/gammabatch_normalization_102/beta#batch_normalization_102/moving_mean'batch_normalization_102/moving_varianceconv2d_transpose_82/kernelconv2d_transpose_82/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_19455715??
?E
?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454205

inputs&
dense_40_19454097:???!
dense_40_19454099:
??.
batch_normalization_99_19454136:	?.
batch_normalization_99_19454138:	?.
batch_normalization_99_19454140:	?.
batch_normalization_99_19454142:	?7
conv2d_transpose_80_19454145:@?*
conv2d_transpose_80_19454147:@.
 batch_normalization_100_19454150:@.
 batch_normalization_100_19454152:@.
 batch_normalization_100_19454154:@.
 batch_normalization_100_19454156:@-
conv2d_112_19454171:@@!
conv2d_112_19454173:@.
 batch_normalization_101_19454176:@.
 batch_normalization_101_19454178:@.
 batch_normalization_101_19454180:@.
 batch_normalization_101_19454182:@6
conv2d_transpose_81_19454185: @*
conv2d_transpose_81_19454187: .
 batch_normalization_102_19454190: .
 batch_normalization_102_19454192: .
 batch_normalization_102_19454194: .
 batch_normalization_102_19454196: 6
conv2d_transpose_82_19454199: *
conv2d_transpose_82_19454201:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?"conv2d_112/StatefulPartitionedCall?+conv2d_transpose_80/StatefulPartitionedCall?+conv2d_transpose_81/StatefulPartitionedCall?+conv2d_transpose_82/StatefulPartitionedCall? dense_40/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_19454097dense_40_19454099*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_40_layer_call_and_return_conditional_losses_194540962"
 dense_40/StatefulPartitionedCall?
reshape_20/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_194541162
reshape_20/PartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall#reshape_20/PartitionedCall:output:0batch_normalization_99_19454136batch_normalization_99_19454138batch_normalization_99_19454140batch_normalization_99_19454142*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_1945413520
.batch_normalization_99/StatefulPartitionedCall?
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0conv2d_transpose_80_19454145conv2d_transpose_80_19454147*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_194536012-
+conv2d_transpose_80/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_80/StatefulPartitionedCall:output:0 batch_normalization_100_19454150 batch_normalization_100_19454152 batch_normalization_100_19454154 batch_normalization_100_19454156*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_1945363321
/batch_normalization_100/StatefulPartitionedCall?
"conv2d_112/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0conv2d_112_19454171conv2d_112_19454173*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_112_layer_call_and_return_conditional_losses_194541702$
"conv2d_112/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall+conv2d_112/StatefulPartitionedCall:output:0 batch_normalization_101_19454176 batch_normalization_101_19454178 batch_normalization_101_19454180 batch_normalization_101_19454182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_1945375921
/batch_normalization_101/StatefulPartitionedCall?
+conv2d_transpose_81/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0conv2d_transpose_81_19454185conv2d_transpose_81_19454187*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_194538982-
+conv2d_transpose_81/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_81/StatefulPartitionedCall:output:0 batch_normalization_102_19454190 batch_normalization_102_19454192 batch_normalization_102_19454194 batch_normalization_102_19454196*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_1945393021
/batch_normalization_102/StatefulPartitionedCall?
+conv2d_transpose_82/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0conv2d_transpose_82_19454199conv2d_transpose_82_19454201*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_194540692-
+conv2d_transpose_82/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_82/StatefulPartitionedCall:output:00^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall#^conv2d_112/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall,^conv2d_transpose_81/StatefulPartitionedCall,^conv2d_transpose_82/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2H
"conv2d_112/StatefulPartitionedCall"conv2d_112/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2Z
+conv2d_transpose_81/StatefulPartitionedCall+conv2d_transpose_81/StatefulPartitionedCall2Z
+conv2d_transpose_82/StatefulPartitionedCall+conv2d_transpose_82/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455266

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_40_layer_call_and_return_conditional_losses_19454096

inputs3
matmul_readvariableop_resource:???/
biasadd_readvariableop_resource:
??
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:??*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455320

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_112_layer_call_and_return_conditional_losses_19455402

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19453462

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19455446

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
0__inference_sequential_60_layer_call_fn_19454260
dense_40_input
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?$
	unknown_5:@?
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17: @

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_60_layer_call_and_return_conditional_losses_194542052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_40_input
?
?
6__inference_conv2d_transpose_80_layer_call_fn_19453611

inputs"
unknown:@?
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_194536012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_99_layer_call_fn_19455248

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_194543032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_102_layer_call_fn_19455490

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_194539742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19453930

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19455364

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_99_layer_call_fn_19455222

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_194535062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_99_layer_call_fn_19455235

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_194541352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19454135

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_sequential_60_layer_call_fn_19454872

inputs
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?$
	unknown_5:@?
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17: @

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_60_layer_call_and_return_conditional_losses_194544552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_reshape_20_layer_call_and_return_conditional_losses_19454116

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19454303

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_112_layer_call_fn_19455391

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_112_layer_call_and_return_conditional_losses_194541702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455302

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19453633

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19455158

inputs<
'dense_40_matmul_readvariableop_resource:???8
(dense_40_biasadd_readvariableop_resource:
??=
.batch_normalization_99_readvariableop_resource:	??
0batch_normalization_99_readvariableop_1_resource:	?N
?batch_normalization_99_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_80_conv2d_transpose_readvariableop_resource:@?A
3conv2d_transpose_80_biasadd_readvariableop_resource:@=
/batch_normalization_100_readvariableop_resource:@?
1batch_normalization_100_readvariableop_1_resource:@N
@batch_normalization_100_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource:@C
)conv2d_112_conv2d_readvariableop_resource:@@8
*conv2d_112_biasadd_readvariableop_resource:@=
/batch_normalization_101_readvariableop_resource:@?
1batch_normalization_101_readvariableop_1_resource:@N
@batch_normalization_101_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_81_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_81_biasadd_readvariableop_resource: =
/batch_normalization_102_readvariableop_resource: ?
1batch_normalization_102_readvariableop_1_resource: N
@batch_normalization_102_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_82_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_82_biasadd_readvariableop_resource:
identity??&batch_normalization_100/AssignNewValue?(batch_normalization_100/AssignNewValue_1?7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_100/ReadVariableOp?(batch_normalization_100/ReadVariableOp_1?&batch_normalization_101/AssignNewValue?(batch_normalization_101/AssignNewValue_1?7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_101/ReadVariableOp?(batch_normalization_101/ReadVariableOp_1?&batch_normalization_102/AssignNewValue?(batch_normalization_102/AssignNewValue_1?7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_102/ReadVariableOp?(batch_normalization_102/ReadVariableOp_1?%batch_normalization_99/AssignNewValue?'batch_normalization_99/AssignNewValue_1?6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_99/ReadVariableOp?'batch_normalization_99/ReadVariableOp_1?!conv2d_112/BiasAdd/ReadVariableOp? conv2d_112/Conv2D/ReadVariableOp?*conv2d_transpose_80/BiasAdd/ReadVariableOp?3conv2d_transpose_80/conv2d_transpose/ReadVariableOp?*conv2d_transpose_81/BiasAdd/ReadVariableOp?3conv2d_transpose_81/conv2d_transpose/ReadVariableOp?*conv2d_transpose_82/BiasAdd/ReadVariableOp?3conv2d_transpose_82/conv2d_transpose/ReadVariableOp?dense_40/BiasAdd/ReadVariableOp?dense_40/MatMul/ReadVariableOp?
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02 
dense_40/MatMul/ReadVariableOp?
dense_40/MatMulMatMulinputs&dense_40/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_40/MatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_40/BiasAddm
reshape_20/ShapeShapedense_40/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_20/Shape?
reshape_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_20/strided_slice/stack?
 reshape_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_20/strided_slice/stack_1?
 reshape_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_20/strided_slice/stack_2?
reshape_20/strided_sliceStridedSlicereshape_20/Shape:output:0'reshape_20/strided_slice/stack:output:0)reshape_20/strided_slice/stack_1:output:0)reshape_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_20/strided_slicez
reshape_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_20/Reshape/shape/1z
reshape_20/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_20/Reshape/shape/2{
reshape_20/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_20/Reshape/shape/3?
reshape_20/Reshape/shapePack!reshape_20/strided_slice:output:0#reshape_20/Reshape/shape/1:output:0#reshape_20/Reshape/shape/2:output:0#reshape_20/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_20/Reshape/shape?
reshape_20/ReshapeReshapedense_40/BiasAdd:output:0!reshape_20/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_20/Reshape?
%batch_normalization_99/ReadVariableOpReadVariableOp.batch_normalization_99_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_99/ReadVariableOp?
'batch_normalization_99/ReadVariableOp_1ReadVariableOp0batch_normalization_99_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_99/ReadVariableOp_1?
6batch_normalization_99/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_99_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_99/FusedBatchNormV3FusedBatchNormV3reshape_20/Reshape:output:0-batch_normalization_99/ReadVariableOp:value:0/batch_normalization_99/ReadVariableOp_1:value:0>batch_normalization_99/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_99/FusedBatchNormV3?
%batch_normalization_99/AssignNewValueAssignVariableOp?batch_normalization_99_fusedbatchnormv3_readvariableop_resource4batch_normalization_99/FusedBatchNormV3:batch_mean:07^batch_normalization_99/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_99/AssignNewValue?
'batch_normalization_99/AssignNewValue_1AssignVariableOpAbatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_99/FusedBatchNormV3:batch_variance:09^batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_99/AssignNewValue_1?
conv2d_transpose_80/ShapeShape+batch_normalization_99/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_80/Shape?
'conv2d_transpose_80/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_80/strided_slice/stack?
)conv2d_transpose_80/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_1?
)conv2d_transpose_80/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_2?
!conv2d_transpose_80/strided_sliceStridedSlice"conv2d_transpose_80/Shape:output:00conv2d_transpose_80/strided_slice/stack:output:02conv2d_transpose_80/strided_slice/stack_1:output:02conv2d_transpose_80/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_80/strided_slice|
conv2d_transpose_80/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_80/stack/1|
conv2d_transpose_80/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_80/stack/2|
conv2d_transpose_80/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_80/stack/3?
conv2d_transpose_80/stackPack*conv2d_transpose_80/strided_slice:output:0$conv2d_transpose_80/stack/1:output:0$conv2d_transpose_80/stack/2:output:0$conv2d_transpose_80/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_80/stack?
)conv2d_transpose_80/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_80/strided_slice_1/stack?
+conv2d_transpose_80/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_1?
+conv2d_transpose_80/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_2?
#conv2d_transpose_80/strided_slice_1StridedSlice"conv2d_transpose_80/stack:output:02conv2d_transpose_80/strided_slice_1/stack:output:04conv2d_transpose_80/strided_slice_1/stack_1:output:04conv2d_transpose_80/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_1?
3conv2d_transpose_80/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_80_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_80/conv2d_transposeConv2DBackpropInput"conv2d_transpose_80/stack:output:0;conv2d_transpose_80/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_99/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2&
$conv2d_transpose_80/conv2d_transpose?
*conv2d_transpose_80/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_80_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_80/BiasAdd/ReadVariableOp?
conv2d_transpose_80/BiasAddBiasAdd-conv2d_transpose_80/conv2d_transpose:output:02conv2d_transpose_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_80/BiasAdd?
conv2d_transpose_80/SeluSelu$conv2d_transpose_80/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_80/Selu?
&batch_normalization_100/ReadVariableOpReadVariableOp/batch_normalization_100_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_100/ReadVariableOp?
(batch_normalization_100/ReadVariableOp_1ReadVariableOp1batch_normalization_100_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_100/ReadVariableOp_1?
7batch_normalization_100/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_100_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_100/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_80/Selu:activations:0.batch_normalization_100/ReadVariableOp:value:00batch_normalization_100/ReadVariableOp_1:value:0?batch_normalization_100/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_100/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_100/FusedBatchNormV3?
&batch_normalization_100/AssignNewValueAssignVariableOp@batch_normalization_100_fusedbatchnormv3_readvariableop_resource5batch_normalization_100/FusedBatchNormV3:batch_mean:08^batch_normalization_100/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_100/AssignNewValue?
(batch_normalization_100/AssignNewValue_1AssignVariableOpBbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_100/FusedBatchNormV3:batch_variance:0:^batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_100/AssignNewValue_1?
 conv2d_112/Conv2D/ReadVariableOpReadVariableOp)conv2d_112_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 conv2d_112/Conv2D/ReadVariableOp?
conv2d_112/Conv2DConv2D,batch_normalization_100/FusedBatchNormV3:y:0(conv2d_112/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_112/Conv2D?
!conv2d_112/BiasAdd/ReadVariableOpReadVariableOp*conv2d_112_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_112/BiasAdd/ReadVariableOp?
conv2d_112/BiasAddBiasAddconv2d_112/Conv2D:output:0)conv2d_112/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_112/BiasAdd?
conv2d_112/SeluSeluconv2d_112/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_112/Selu?
&batch_normalization_101/ReadVariableOpReadVariableOp/batch_normalization_101_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_101/ReadVariableOp?
(batch_normalization_101/ReadVariableOp_1ReadVariableOp1batch_normalization_101_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_101/ReadVariableOp_1?
7batch_normalization_101/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_101_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_101/FusedBatchNormV3FusedBatchNormV3conv2d_112/Selu:activations:0.batch_normalization_101/ReadVariableOp:value:00batch_normalization_101/ReadVariableOp_1:value:0?batch_normalization_101/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_101/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_101/FusedBatchNormV3?
&batch_normalization_101/AssignNewValueAssignVariableOp@batch_normalization_101_fusedbatchnormv3_readvariableop_resource5batch_normalization_101/FusedBatchNormV3:batch_mean:08^batch_normalization_101/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_101/AssignNewValue?
(batch_normalization_101/AssignNewValue_1AssignVariableOpBbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_101/FusedBatchNormV3:batch_variance:0:^batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_101/AssignNewValue_1?
conv2d_transpose_81/ShapeShape,batch_normalization_101/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_81/Shape?
'conv2d_transpose_81/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_81/strided_slice/stack?
)conv2d_transpose_81/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_81/strided_slice/stack_1?
)conv2d_transpose_81/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_81/strided_slice/stack_2?
!conv2d_transpose_81/strided_sliceStridedSlice"conv2d_transpose_81/Shape:output:00conv2d_transpose_81/strided_slice/stack:output:02conv2d_transpose_81/strided_slice/stack_1:output:02conv2d_transpose_81/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_81/strided_slice|
conv2d_transpose_81/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_81/stack/1|
conv2d_transpose_81/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_81/stack/2|
conv2d_transpose_81/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_81/stack/3?
conv2d_transpose_81/stackPack*conv2d_transpose_81/strided_slice:output:0$conv2d_transpose_81/stack/1:output:0$conv2d_transpose_81/stack/2:output:0$conv2d_transpose_81/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_81/stack?
)conv2d_transpose_81/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_81/strided_slice_1/stack?
+conv2d_transpose_81/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_81/strided_slice_1/stack_1?
+conv2d_transpose_81/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_81/strided_slice_1/stack_2?
#conv2d_transpose_81/strided_slice_1StridedSlice"conv2d_transpose_81/stack:output:02conv2d_transpose_81/strided_slice_1/stack:output:04conv2d_transpose_81/strided_slice_1/stack_1:output:04conv2d_transpose_81/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_81/strided_slice_1?
3conv2d_transpose_81/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_81_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_81/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_81/conv2d_transposeConv2DBackpropInput"conv2d_transpose_81/stack:output:0;conv2d_transpose_81/conv2d_transpose/ReadVariableOp:value:0,batch_normalization_101/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2&
$conv2d_transpose_81/conv2d_transpose?
*conv2d_transpose_81/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_81/BiasAdd/ReadVariableOp?
conv2d_transpose_81/BiasAddBiasAdd-conv2d_transpose_81/conv2d_transpose:output:02conv2d_transpose_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_81/BiasAdd?
conv2d_transpose_81/SeluSelu$conv2d_transpose_81/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_81/Selu?
&batch_normalization_102/ReadVariableOpReadVariableOp/batch_normalization_102_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_102/ReadVariableOp?
(batch_normalization_102/ReadVariableOp_1ReadVariableOp1batch_normalization_102_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_102/ReadVariableOp_1?
7batch_normalization_102/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_102_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_102/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_81/Selu:activations:0.batch_normalization_102/ReadVariableOp:value:00batch_normalization_102/ReadVariableOp_1:value:0?batch_normalization_102/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_102/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2*
(batch_normalization_102/FusedBatchNormV3?
&batch_normalization_102/AssignNewValueAssignVariableOp@batch_normalization_102_fusedbatchnormv3_readvariableop_resource5batch_normalization_102/FusedBatchNormV3:batch_mean:08^batch_normalization_102/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_102/AssignNewValue?
(batch_normalization_102/AssignNewValue_1AssignVariableOpBbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_102/FusedBatchNormV3:batch_variance:0:^batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02*
(batch_normalization_102/AssignNewValue_1?
conv2d_transpose_82/ShapeShape,batch_normalization_102/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_82/Shape?
'conv2d_transpose_82/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_82/strided_slice/stack?
)conv2d_transpose_82/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_82/strided_slice/stack_1?
)conv2d_transpose_82/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_82/strided_slice/stack_2?
!conv2d_transpose_82/strided_sliceStridedSlice"conv2d_transpose_82/Shape:output:00conv2d_transpose_82/strided_slice/stack:output:02conv2d_transpose_82/strided_slice/stack_1:output:02conv2d_transpose_82/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_82/strided_slice}
conv2d_transpose_82/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_82/stack/1}
conv2d_transpose_82/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_82/stack/2|
conv2d_transpose_82/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_82/stack/3?
conv2d_transpose_82/stackPack*conv2d_transpose_82/strided_slice:output:0$conv2d_transpose_82/stack/1:output:0$conv2d_transpose_82/stack/2:output:0$conv2d_transpose_82/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_82/stack?
)conv2d_transpose_82/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_82/strided_slice_1/stack?
+conv2d_transpose_82/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_82/strided_slice_1/stack_1?
+conv2d_transpose_82/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_82/strided_slice_1/stack_2?
#conv2d_transpose_82/strided_slice_1StridedSlice"conv2d_transpose_82/stack:output:02conv2d_transpose_82/strided_slice_1/stack:output:04conv2d_transpose_82/strided_slice_1/stack_1:output:04conv2d_transpose_82/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_82/strided_slice_1?
3conv2d_transpose_82/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_82_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_82/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_82/conv2d_transposeConv2DBackpropInput"conv2d_transpose_82/stack:output:0;conv2d_transpose_82/conv2d_transpose/ReadVariableOp:value:0,batch_normalization_102/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$conv2d_transpose_82/conv2d_transpose?
*conv2d_transpose_82/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_82/BiasAdd/ReadVariableOp?
conv2d_transpose_82/BiasAddBiasAdd-conv2d_transpose_82/conv2d_transpose:output:02conv2d_transpose_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_82/BiasAdd?
conv2d_transpose_82/TanhTanh$conv2d_transpose_82/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_82/Tanh?
IdentityIdentityconv2d_transpose_82/Tanh:y:0'^batch_normalization_100/AssignNewValue)^batch_normalization_100/AssignNewValue_18^batch_normalization_100/FusedBatchNormV3/ReadVariableOp:^batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_100/ReadVariableOp)^batch_normalization_100/ReadVariableOp_1'^batch_normalization_101/AssignNewValue)^batch_normalization_101/AssignNewValue_18^batch_normalization_101/FusedBatchNormV3/ReadVariableOp:^batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_101/ReadVariableOp)^batch_normalization_101/ReadVariableOp_1'^batch_normalization_102/AssignNewValue)^batch_normalization_102/AssignNewValue_18^batch_normalization_102/FusedBatchNormV3/ReadVariableOp:^batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_102/ReadVariableOp)^batch_normalization_102/ReadVariableOp_1&^batch_normalization_99/AssignNewValue(^batch_normalization_99/AssignNewValue_17^batch_normalization_99/FusedBatchNormV3/ReadVariableOp9^batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_99/ReadVariableOp(^batch_normalization_99/ReadVariableOp_1"^conv2d_112/BiasAdd/ReadVariableOp!^conv2d_112/Conv2D/ReadVariableOp+^conv2d_transpose_80/BiasAdd/ReadVariableOp4^conv2d_transpose_80/conv2d_transpose/ReadVariableOp+^conv2d_transpose_81/BiasAdd/ReadVariableOp4^conv2d_transpose_81/conv2d_transpose/ReadVariableOp+^conv2d_transpose_82/BiasAdd/ReadVariableOp4^conv2d_transpose_82/conv2d_transpose/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_100/AssignNewValue&batch_normalization_100/AssignNewValue2T
(batch_normalization_100/AssignNewValue_1(batch_normalization_100/AssignNewValue_12r
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp7batch_normalization_100/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_19batch_normalization_100/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_100/ReadVariableOp&batch_normalization_100/ReadVariableOp2T
(batch_normalization_100/ReadVariableOp_1(batch_normalization_100/ReadVariableOp_12P
&batch_normalization_101/AssignNewValue&batch_normalization_101/AssignNewValue2T
(batch_normalization_101/AssignNewValue_1(batch_normalization_101/AssignNewValue_12r
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp7batch_normalization_101/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_19batch_normalization_101/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_101/ReadVariableOp&batch_normalization_101/ReadVariableOp2T
(batch_normalization_101/ReadVariableOp_1(batch_normalization_101/ReadVariableOp_12P
&batch_normalization_102/AssignNewValue&batch_normalization_102/AssignNewValue2T
(batch_normalization_102/AssignNewValue_1(batch_normalization_102/AssignNewValue_12r
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp7batch_normalization_102/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_19batch_normalization_102/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_102/ReadVariableOp&batch_normalization_102/ReadVariableOp2T
(batch_normalization_102/ReadVariableOp_1(batch_normalization_102/ReadVariableOp_12N
%batch_normalization_99/AssignNewValue%batch_normalization_99/AssignNewValue2R
'batch_normalization_99/AssignNewValue_1'batch_normalization_99/AssignNewValue_12p
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp6batch_normalization_99/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_18batch_normalization_99/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_99/ReadVariableOp%batch_normalization_99/ReadVariableOp2R
'batch_normalization_99/ReadVariableOp_1'batch_normalization_99/ReadVariableOp_12F
!conv2d_112/BiasAdd/ReadVariableOp!conv2d_112/BiasAdd/ReadVariableOp2D
 conv2d_112/Conv2D/ReadVariableOp conv2d_112/Conv2D/ReadVariableOp2X
*conv2d_transpose_80/BiasAdd/ReadVariableOp*conv2d_transpose_80/BiasAdd/ReadVariableOp2j
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp3conv2d_transpose_80/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_81/BiasAdd/ReadVariableOp*conv2d_transpose_81/BiasAdd/ReadVariableOp2j
3conv2d_transpose_81/conv2d_transpose/ReadVariableOp3conv2d_transpose_81/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_82/BiasAdd/ReadVariableOp*conv2d_transpose_82/BiasAdd/ReadVariableOp2j
3conv2d_transpose_82/conv2d_transpose/ReadVariableOp3conv2d_transpose_82/conv2d_transpose/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455284

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19453803

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_19454758
dense_40_input
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?$
	unknown_5:@?
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17: @

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_194534402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_40_input
?
?
:__inference_batch_normalization_100_layer_call_fn_19455333

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_194536332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19455464

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19453974

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?t
?
$__inference__traced_restore_19455715
file_prefix5
 assignvariableop_dense_40_kernel:???0
 assignvariableop_1_dense_40_bias:
??>
/assignvariableop_2_batch_normalization_99_gamma:	?=
.assignvariableop_3_batch_normalization_99_beta:	?D
5assignvariableop_4_batch_normalization_99_moving_mean:	?H
9assignvariableop_5_batch_normalization_99_moving_variance:	?H
-assignvariableop_6_conv2d_transpose_80_kernel:@?9
+assignvariableop_7_conv2d_transpose_80_bias:@>
0assignvariableop_8_batch_normalization_100_gamma:@=
/assignvariableop_9_batch_normalization_100_beta:@E
7assignvariableop_10_batch_normalization_100_moving_mean:@I
;assignvariableop_11_batch_normalization_100_moving_variance:@?
%assignvariableop_12_conv2d_112_kernel:@@1
#assignvariableop_13_conv2d_112_bias:@?
1assignvariableop_14_batch_normalization_101_gamma:@>
0assignvariableop_15_batch_normalization_101_beta:@E
7assignvariableop_16_batch_normalization_101_moving_mean:@I
;assignvariableop_17_batch_normalization_101_moving_variance:@H
.assignvariableop_18_conv2d_transpose_81_kernel: @:
,assignvariableop_19_conv2d_transpose_81_bias: ?
1assignvariableop_20_batch_normalization_102_gamma: >
0assignvariableop_21_batch_normalization_102_beta: E
7assignvariableop_22_batch_normalization_102_moving_mean: I
;assignvariableop_23_batch_normalization_102_moving_variance: H
.assignvariableop_24_conv2d_transpose_82_kernel: :
,assignvariableop_25_conv2d_transpose_82_bias:
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_40_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_40_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_99_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_99_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_99_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_99_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_80_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_80_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_100_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_100_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_100_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_100_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_conv2d_112_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_112_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batch_normalization_101_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_101_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batch_normalization_101_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batch_normalization_101_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_conv2d_transpose_81_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_conv2d_transpose_81_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_batch_normalization_102_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_102_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_102_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp;assignvariableop_23_batch_normalization_102_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_conv2d_transpose_82_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_conv2d_transpose_82_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26?
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_27"#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
I
-__inference_reshape_20_layer_call_fn_19455182

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_194541162
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_112_layer_call_and_return_conditional_losses_19454170

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19455015

inputs<
'dense_40_matmul_readvariableop_resource:???8
(dense_40_biasadd_readvariableop_resource:
??=
.batch_normalization_99_readvariableop_resource:	??
0batch_normalization_99_readvariableop_1_resource:	?N
?batch_normalization_99_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_80_conv2d_transpose_readvariableop_resource:@?A
3conv2d_transpose_80_biasadd_readvariableop_resource:@=
/batch_normalization_100_readvariableop_resource:@?
1batch_normalization_100_readvariableop_1_resource:@N
@batch_normalization_100_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource:@C
)conv2d_112_conv2d_readvariableop_resource:@@8
*conv2d_112_biasadd_readvariableop_resource:@=
/batch_normalization_101_readvariableop_resource:@?
1batch_normalization_101_readvariableop_1_resource:@N
@batch_normalization_101_fusedbatchnormv3_readvariableop_resource:@P
Bbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_81_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_81_biasadd_readvariableop_resource: =
/batch_normalization_102_readvariableop_resource: ?
1batch_normalization_102_readvariableop_1_resource: N
@batch_normalization_102_fusedbatchnormv3_readvariableop_resource: P
Bbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_82_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_82_biasadd_readvariableop_resource:
identity??7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_100/ReadVariableOp?(batch_normalization_100/ReadVariableOp_1?7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_101/ReadVariableOp?(batch_normalization_101/ReadVariableOp_1?7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_102/ReadVariableOp?(batch_normalization_102/ReadVariableOp_1?6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_99/ReadVariableOp?'batch_normalization_99/ReadVariableOp_1?!conv2d_112/BiasAdd/ReadVariableOp? conv2d_112/Conv2D/ReadVariableOp?*conv2d_transpose_80/BiasAdd/ReadVariableOp?3conv2d_transpose_80/conv2d_transpose/ReadVariableOp?*conv2d_transpose_81/BiasAdd/ReadVariableOp?3conv2d_transpose_81/conv2d_transpose/ReadVariableOp?*conv2d_transpose_82/BiasAdd/ReadVariableOp?3conv2d_transpose_82/conv2d_transpose/ReadVariableOp?dense_40/BiasAdd/ReadVariableOp?dense_40/MatMul/ReadVariableOp?
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02 
dense_40/MatMul/ReadVariableOp?
dense_40/MatMulMatMulinputs&dense_40/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_40/MatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_40/BiasAddm
reshape_20/ShapeShapedense_40/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_20/Shape?
reshape_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_20/strided_slice/stack?
 reshape_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_20/strided_slice/stack_1?
 reshape_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_20/strided_slice/stack_2?
reshape_20/strided_sliceStridedSlicereshape_20/Shape:output:0'reshape_20/strided_slice/stack:output:0)reshape_20/strided_slice/stack_1:output:0)reshape_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_20/strided_slicez
reshape_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_20/Reshape/shape/1z
reshape_20/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_20/Reshape/shape/2{
reshape_20/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_20/Reshape/shape/3?
reshape_20/Reshape/shapePack!reshape_20/strided_slice:output:0#reshape_20/Reshape/shape/1:output:0#reshape_20/Reshape/shape/2:output:0#reshape_20/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_20/Reshape/shape?
reshape_20/ReshapeReshapedense_40/BiasAdd:output:0!reshape_20/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_20/Reshape?
%batch_normalization_99/ReadVariableOpReadVariableOp.batch_normalization_99_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_99/ReadVariableOp?
'batch_normalization_99/ReadVariableOp_1ReadVariableOp0batch_normalization_99_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_99/ReadVariableOp_1?
6batch_normalization_99/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_99_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_99_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_99/FusedBatchNormV3FusedBatchNormV3reshape_20/Reshape:output:0-batch_normalization_99/ReadVariableOp:value:0/batch_normalization_99/ReadVariableOp_1:value:0>batch_normalization_99/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_99/FusedBatchNormV3?
conv2d_transpose_80/ShapeShape+batch_normalization_99/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_80/Shape?
'conv2d_transpose_80/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_80/strided_slice/stack?
)conv2d_transpose_80/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_1?
)conv2d_transpose_80/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_80/strided_slice/stack_2?
!conv2d_transpose_80/strided_sliceStridedSlice"conv2d_transpose_80/Shape:output:00conv2d_transpose_80/strided_slice/stack:output:02conv2d_transpose_80/strided_slice/stack_1:output:02conv2d_transpose_80/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_80/strided_slice|
conv2d_transpose_80/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_80/stack/1|
conv2d_transpose_80/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_80/stack/2|
conv2d_transpose_80/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_80/stack/3?
conv2d_transpose_80/stackPack*conv2d_transpose_80/strided_slice:output:0$conv2d_transpose_80/stack/1:output:0$conv2d_transpose_80/stack/2:output:0$conv2d_transpose_80/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_80/stack?
)conv2d_transpose_80/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_80/strided_slice_1/stack?
+conv2d_transpose_80/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_1?
+conv2d_transpose_80/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_80/strided_slice_1/stack_2?
#conv2d_transpose_80/strided_slice_1StridedSlice"conv2d_transpose_80/stack:output:02conv2d_transpose_80/strided_slice_1/stack:output:04conv2d_transpose_80/strided_slice_1/stack_1:output:04conv2d_transpose_80/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_80/strided_slice_1?
3conv2d_transpose_80/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_80_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_80/conv2d_transposeConv2DBackpropInput"conv2d_transpose_80/stack:output:0;conv2d_transpose_80/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_99/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2&
$conv2d_transpose_80/conv2d_transpose?
*conv2d_transpose_80/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_80_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_80/BiasAdd/ReadVariableOp?
conv2d_transpose_80/BiasAddBiasAdd-conv2d_transpose_80/conv2d_transpose:output:02conv2d_transpose_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_80/BiasAdd?
conv2d_transpose_80/SeluSelu$conv2d_transpose_80/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_80/Selu?
&batch_normalization_100/ReadVariableOpReadVariableOp/batch_normalization_100_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_100/ReadVariableOp?
(batch_normalization_100/ReadVariableOp_1ReadVariableOp1batch_normalization_100_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_100/ReadVariableOp_1?
7batch_normalization_100/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_100_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_100_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_100/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_80/Selu:activations:0.batch_normalization_100/ReadVariableOp:value:00batch_normalization_100/ReadVariableOp_1:value:0?batch_normalization_100/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_100/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2*
(batch_normalization_100/FusedBatchNormV3?
 conv2d_112/Conv2D/ReadVariableOpReadVariableOp)conv2d_112_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 conv2d_112/Conv2D/ReadVariableOp?
conv2d_112/Conv2DConv2D,batch_normalization_100/FusedBatchNormV3:y:0(conv2d_112/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2
conv2d_112/Conv2D?
!conv2d_112/BiasAdd/ReadVariableOpReadVariableOp*conv2d_112_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_112/BiasAdd/ReadVariableOp?
conv2d_112/BiasAddBiasAddconv2d_112/Conv2D:output:0)conv2d_112/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_112/BiasAdd?
conv2d_112/SeluSeluconv2d_112/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_112/Selu?
&batch_normalization_101/ReadVariableOpReadVariableOp/batch_normalization_101_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_101/ReadVariableOp?
(batch_normalization_101/ReadVariableOp_1ReadVariableOp1batch_normalization_101_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_101/ReadVariableOp_1?
7batch_normalization_101/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_101_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_101_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_101/FusedBatchNormV3FusedBatchNormV3conv2d_112/Selu:activations:0.batch_normalization_101/ReadVariableOp:value:00batch_normalization_101/ReadVariableOp_1:value:0?batch_normalization_101/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_101/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2*
(batch_normalization_101/FusedBatchNormV3?
conv2d_transpose_81/ShapeShape,batch_normalization_101/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_81/Shape?
'conv2d_transpose_81/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_81/strided_slice/stack?
)conv2d_transpose_81/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_81/strided_slice/stack_1?
)conv2d_transpose_81/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_81/strided_slice/stack_2?
!conv2d_transpose_81/strided_sliceStridedSlice"conv2d_transpose_81/Shape:output:00conv2d_transpose_81/strided_slice/stack:output:02conv2d_transpose_81/strided_slice/stack_1:output:02conv2d_transpose_81/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_81/strided_slice|
conv2d_transpose_81/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_81/stack/1|
conv2d_transpose_81/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_81/stack/2|
conv2d_transpose_81/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_81/stack/3?
conv2d_transpose_81/stackPack*conv2d_transpose_81/strided_slice:output:0$conv2d_transpose_81/stack/1:output:0$conv2d_transpose_81/stack/2:output:0$conv2d_transpose_81/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_81/stack?
)conv2d_transpose_81/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_81/strided_slice_1/stack?
+conv2d_transpose_81/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_81/strided_slice_1/stack_1?
+conv2d_transpose_81/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_81/strided_slice_1/stack_2?
#conv2d_transpose_81/strided_slice_1StridedSlice"conv2d_transpose_81/stack:output:02conv2d_transpose_81/strided_slice_1/stack:output:04conv2d_transpose_81/strided_slice_1/stack_1:output:04conv2d_transpose_81/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_81/strided_slice_1?
3conv2d_transpose_81/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_81_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_81/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_81/conv2d_transposeConv2DBackpropInput"conv2d_transpose_81/stack:output:0;conv2d_transpose_81/conv2d_transpose/ReadVariableOp:value:0,batch_normalization_101/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2&
$conv2d_transpose_81/conv2d_transpose?
*conv2d_transpose_81/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_81/BiasAdd/ReadVariableOp?
conv2d_transpose_81/BiasAddBiasAdd-conv2d_transpose_81/conv2d_transpose:output:02conv2d_transpose_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_81/BiasAdd?
conv2d_transpose_81/SeluSelu$conv2d_transpose_81/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_81/Selu?
&batch_normalization_102/ReadVariableOpReadVariableOp/batch_normalization_102_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_102/ReadVariableOp?
(batch_normalization_102/ReadVariableOp_1ReadVariableOp1batch_normalization_102_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_102/ReadVariableOp_1?
7batch_normalization_102/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_102_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp?
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_102_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?
(batch_normalization_102/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_81/Selu:activations:0.batch_normalization_102/ReadVariableOp:value:00batch_normalization_102/ReadVariableOp_1:value:0?batch_normalization_102/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_102/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 2*
(batch_normalization_102/FusedBatchNormV3?
conv2d_transpose_82/ShapeShape,batch_normalization_102/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_82/Shape?
'conv2d_transpose_82/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_82/strided_slice/stack?
)conv2d_transpose_82/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_82/strided_slice/stack_1?
)conv2d_transpose_82/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_82/strided_slice/stack_2?
!conv2d_transpose_82/strided_sliceStridedSlice"conv2d_transpose_82/Shape:output:00conv2d_transpose_82/strided_slice/stack:output:02conv2d_transpose_82/strided_slice/stack_1:output:02conv2d_transpose_82/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_82/strided_slice}
conv2d_transpose_82/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_82/stack/1}
conv2d_transpose_82/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_82/stack/2|
conv2d_transpose_82/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_82/stack/3?
conv2d_transpose_82/stackPack*conv2d_transpose_82/strided_slice:output:0$conv2d_transpose_82/stack/1:output:0$conv2d_transpose_82/stack/2:output:0$conv2d_transpose_82/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_82/stack?
)conv2d_transpose_82/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_82/strided_slice_1/stack?
+conv2d_transpose_82/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_82/strided_slice_1/stack_1?
+conv2d_transpose_82/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_82/strided_slice_1/stack_2?
#conv2d_transpose_82/strided_slice_1StridedSlice"conv2d_transpose_82/stack:output:02conv2d_transpose_82/strided_slice_1/stack:output:04conv2d_transpose_82/strided_slice_1/stack_1:output:04conv2d_transpose_82/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_82/strided_slice_1?
3conv2d_transpose_82/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_82_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_82/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_82/conv2d_transposeConv2DBackpropInput"conv2d_transpose_82/stack:output:0;conv2d_transpose_82/conv2d_transpose/ReadVariableOp:value:0,batch_normalization_102/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$conv2d_transpose_82/conv2d_transpose?
*conv2d_transpose_82/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_82/BiasAdd/ReadVariableOp?
conv2d_transpose_82/BiasAddBiasAdd-conv2d_transpose_82/conv2d_transpose:output:02conv2d_transpose_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_82/BiasAdd?
conv2d_transpose_82/TanhTanh$conv2d_transpose_82/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_82/Tanh?

IdentityIdentityconv2d_transpose_82/Tanh:y:08^batch_normalization_100/FusedBatchNormV3/ReadVariableOp:^batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_100/ReadVariableOp)^batch_normalization_100/ReadVariableOp_18^batch_normalization_101/FusedBatchNormV3/ReadVariableOp:^batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_101/ReadVariableOp)^batch_normalization_101/ReadVariableOp_18^batch_normalization_102/FusedBatchNormV3/ReadVariableOp:^batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_102/ReadVariableOp)^batch_normalization_102/ReadVariableOp_17^batch_normalization_99/FusedBatchNormV3/ReadVariableOp9^batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_99/ReadVariableOp(^batch_normalization_99/ReadVariableOp_1"^conv2d_112/BiasAdd/ReadVariableOp!^conv2d_112/Conv2D/ReadVariableOp+^conv2d_transpose_80/BiasAdd/ReadVariableOp4^conv2d_transpose_80/conv2d_transpose/ReadVariableOp+^conv2d_transpose_81/BiasAdd/ReadVariableOp4^conv2d_transpose_81/conv2d_transpose/ReadVariableOp+^conv2d_transpose_82/BiasAdd/ReadVariableOp4^conv2d_transpose_82/conv2d_transpose/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7batch_normalization_100/FusedBatchNormV3/ReadVariableOp7batch_normalization_100/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_100/FusedBatchNormV3/ReadVariableOp_19batch_normalization_100/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_100/ReadVariableOp&batch_normalization_100/ReadVariableOp2T
(batch_normalization_100/ReadVariableOp_1(batch_normalization_100/ReadVariableOp_12r
7batch_normalization_101/FusedBatchNormV3/ReadVariableOp7batch_normalization_101/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_101/FusedBatchNormV3/ReadVariableOp_19batch_normalization_101/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_101/ReadVariableOp&batch_normalization_101/ReadVariableOp2T
(batch_normalization_101/ReadVariableOp_1(batch_normalization_101/ReadVariableOp_12r
7batch_normalization_102/FusedBatchNormV3/ReadVariableOp7batch_normalization_102/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_102/FusedBatchNormV3/ReadVariableOp_19batch_normalization_102/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_102/ReadVariableOp&batch_normalization_102/ReadVariableOp2T
(batch_normalization_102/ReadVariableOp_1(batch_normalization_102/ReadVariableOp_12p
6batch_normalization_99/FusedBatchNormV3/ReadVariableOp6batch_normalization_99/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_99/FusedBatchNormV3/ReadVariableOp_18batch_normalization_99/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_99/ReadVariableOp%batch_normalization_99/ReadVariableOp2R
'batch_normalization_99/ReadVariableOp_1'batch_normalization_99/ReadVariableOp_12F
!conv2d_112/BiasAdd/ReadVariableOp!conv2d_112/BiasAdd/ReadVariableOp2D
 conv2d_112/Conv2D/ReadVariableOp conv2d_112/Conv2D/ReadVariableOp2X
*conv2d_transpose_80/BiasAdd/ReadVariableOp*conv2d_transpose_80/BiasAdd/ReadVariableOp2j
3conv2d_transpose_80/conv2d_transpose/ReadVariableOp3conv2d_transpose_80/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_81/BiasAdd/ReadVariableOp*conv2d_transpose_81/BiasAdd/ReadVariableOp2j
3conv2d_transpose_81/conv2d_transpose/ReadVariableOp3conv2d_transpose_81/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_82/BiasAdd/ReadVariableOp*conv2d_transpose_82/BiasAdd/ReadVariableOp2j
3conv2d_transpose_82/conv2d_transpose/ReadVariableOp3conv2d_transpose_82/conv2d_transpose/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19455382

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
6__inference_conv2d_transpose_81_layer_call_fn_19453908

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_194538982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19453506

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?%
?
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_19453601

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?E
?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454699
dense_40_input&
dense_40_19454636:???!
dense_40_19454638:
??.
batch_normalization_99_19454642:	?.
batch_normalization_99_19454644:	?.
batch_normalization_99_19454646:	?.
batch_normalization_99_19454648:	?7
conv2d_transpose_80_19454651:@?*
conv2d_transpose_80_19454653:@.
 batch_normalization_100_19454656:@.
 batch_normalization_100_19454658:@.
 batch_normalization_100_19454660:@.
 batch_normalization_100_19454662:@-
conv2d_112_19454665:@@!
conv2d_112_19454667:@.
 batch_normalization_101_19454670:@.
 batch_normalization_101_19454672:@.
 batch_normalization_101_19454674:@.
 batch_normalization_101_19454676:@6
conv2d_transpose_81_19454679: @*
conv2d_transpose_81_19454681: .
 batch_normalization_102_19454684: .
 batch_normalization_102_19454686: .
 batch_normalization_102_19454688: .
 batch_normalization_102_19454690: 6
conv2d_transpose_82_19454693: *
conv2d_transpose_82_19454695:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?"conv2d_112/StatefulPartitionedCall?+conv2d_transpose_80/StatefulPartitionedCall?+conv2d_transpose_81/StatefulPartitionedCall?+conv2d_transpose_82/StatefulPartitionedCall? dense_40/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCalldense_40_inputdense_40_19454636dense_40_19454638*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_40_layer_call_and_return_conditional_losses_194540962"
 dense_40/StatefulPartitionedCall?
reshape_20/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_194541162
reshape_20/PartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall#reshape_20/PartitionedCall:output:0batch_normalization_99_19454642batch_normalization_99_19454644batch_normalization_99_19454646batch_normalization_99_19454648*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_1945430320
.batch_normalization_99/StatefulPartitionedCall?
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0conv2d_transpose_80_19454651conv2d_transpose_80_19454653*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_194536012-
+conv2d_transpose_80/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_80/StatefulPartitionedCall:output:0 batch_normalization_100_19454656 batch_normalization_100_19454658 batch_normalization_100_19454660 batch_normalization_100_19454662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_1945367721
/batch_normalization_100/StatefulPartitionedCall?
"conv2d_112/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0conv2d_112_19454665conv2d_112_19454667*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_112_layer_call_and_return_conditional_losses_194541702$
"conv2d_112/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall+conv2d_112/StatefulPartitionedCall:output:0 batch_normalization_101_19454670 batch_normalization_101_19454672 batch_normalization_101_19454674 batch_normalization_101_19454676*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_1945380321
/batch_normalization_101/StatefulPartitionedCall?
+conv2d_transpose_81/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0conv2d_transpose_81_19454679conv2d_transpose_81_19454681*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_194538982-
+conv2d_transpose_81/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_81/StatefulPartitionedCall:output:0 batch_normalization_102_19454684 batch_normalization_102_19454686 batch_normalization_102_19454688 batch_normalization_102_19454690*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_1945397421
/batch_normalization_102/StatefulPartitionedCall?
+conv2d_transpose_82/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0conv2d_transpose_82_19454693conv2d_transpose_82_19454695*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_194540692-
+conv2d_transpose_82/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_82/StatefulPartitionedCall:output:00^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall#^conv2d_112/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall,^conv2d_transpose_81/StatefulPartitionedCall,^conv2d_transpose_82/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2H
"conv2d_112/StatefulPartitionedCall"conv2d_112/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2Z
+conv2d_transpose_81/StatefulPartitionedCall+conv2d_transpose_81/StatefulPartitionedCall2Z
+conv2d_transpose_82/StatefulPartitionedCall+conv2d_transpose_82/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_40_input
?D
?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454455

inputs&
dense_40_19454392:???!
dense_40_19454394:
??.
batch_normalization_99_19454398:	?.
batch_normalization_99_19454400:	?.
batch_normalization_99_19454402:	?.
batch_normalization_99_19454404:	?7
conv2d_transpose_80_19454407:@?*
conv2d_transpose_80_19454409:@.
 batch_normalization_100_19454412:@.
 batch_normalization_100_19454414:@.
 batch_normalization_100_19454416:@.
 batch_normalization_100_19454418:@-
conv2d_112_19454421:@@!
conv2d_112_19454423:@.
 batch_normalization_101_19454426:@.
 batch_normalization_101_19454428:@.
 batch_normalization_101_19454430:@.
 batch_normalization_101_19454432:@6
conv2d_transpose_81_19454435: @*
conv2d_transpose_81_19454437: .
 batch_normalization_102_19454440: .
 batch_normalization_102_19454442: .
 batch_normalization_102_19454444: .
 batch_normalization_102_19454446: 6
conv2d_transpose_82_19454449: *
conv2d_transpose_82_19454451:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?"conv2d_112/StatefulPartitionedCall?+conv2d_transpose_80/StatefulPartitionedCall?+conv2d_transpose_81/StatefulPartitionedCall?+conv2d_transpose_82/StatefulPartitionedCall? dense_40/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_19454392dense_40_19454394*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_40_layer_call_and_return_conditional_losses_194540962"
 dense_40/StatefulPartitionedCall?
reshape_20/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_194541162
reshape_20/PartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall#reshape_20/PartitionedCall:output:0batch_normalization_99_19454398batch_normalization_99_19454400batch_normalization_99_19454402batch_normalization_99_19454404*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_1945430320
.batch_normalization_99/StatefulPartitionedCall?
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0conv2d_transpose_80_19454407conv2d_transpose_80_19454409*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_194536012-
+conv2d_transpose_80/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_80/StatefulPartitionedCall:output:0 batch_normalization_100_19454412 batch_normalization_100_19454414 batch_normalization_100_19454416 batch_normalization_100_19454418*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_1945367721
/batch_normalization_100/StatefulPartitionedCall?
"conv2d_112/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0conv2d_112_19454421conv2d_112_19454423*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_112_layer_call_and_return_conditional_losses_194541702$
"conv2d_112/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall+conv2d_112/StatefulPartitionedCall:output:0 batch_normalization_101_19454426 batch_normalization_101_19454428 batch_normalization_101_19454430 batch_normalization_101_19454432*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_1945380321
/batch_normalization_101/StatefulPartitionedCall?
+conv2d_transpose_81/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0conv2d_transpose_81_19454435conv2d_transpose_81_19454437*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_194538982-
+conv2d_transpose_81/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_81/StatefulPartitionedCall:output:0 batch_normalization_102_19454440 batch_normalization_102_19454442 batch_normalization_102_19454444 batch_normalization_102_19454446*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_1945397421
/batch_normalization_102/StatefulPartitionedCall?
+conv2d_transpose_82/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0conv2d_transpose_82_19454449conv2d_transpose_82_19454451*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_194540692-
+conv2d_transpose_82/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_82/StatefulPartitionedCall:output:00^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall#^conv2d_112/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall,^conv2d_transpose_81/StatefulPartitionedCall,^conv2d_transpose_82/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2H
"conv2d_112/StatefulPartitionedCall"conv2d_112/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2Z
+conv2d_transpose_81/StatefulPartitionedCall+conv2d_transpose_81/StatefulPartitionedCall2Z
+conv2d_transpose_82/StatefulPartitionedCall+conv2d_transpose_82/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_100_layer_call_fn_19455346

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_194536772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?%
?
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_19454069

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_101_layer_call_fn_19455428

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_194538032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19453759

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?%
?
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_19453898

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
SeluSeluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Selu?
IdentityIdentitySelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_19453440
dense_40_inputJ
5sequential_60_dense_40_matmul_readvariableop_resource:???F
6sequential_60_dense_40_biasadd_readvariableop_resource:
??K
<sequential_60_batch_normalization_99_readvariableop_resource:	?M
>sequential_60_batch_normalization_99_readvariableop_1_resource:	?\
Msequential_60_batch_normalization_99_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_60_batch_normalization_99_fusedbatchnormv3_readvariableop_1_resource:	?e
Jsequential_60_conv2d_transpose_80_conv2d_transpose_readvariableop_resource:@?O
Asequential_60_conv2d_transpose_80_biasadd_readvariableop_resource:@K
=sequential_60_batch_normalization_100_readvariableop_resource:@M
?sequential_60_batch_normalization_100_readvariableop_1_resource:@\
Nsequential_60_batch_normalization_100_fusedbatchnormv3_readvariableop_resource:@^
Psequential_60_batch_normalization_100_fusedbatchnormv3_readvariableop_1_resource:@Q
7sequential_60_conv2d_112_conv2d_readvariableop_resource:@@F
8sequential_60_conv2d_112_biasadd_readvariableop_resource:@K
=sequential_60_batch_normalization_101_readvariableop_resource:@M
?sequential_60_batch_normalization_101_readvariableop_1_resource:@\
Nsequential_60_batch_normalization_101_fusedbatchnormv3_readvariableop_resource:@^
Psequential_60_batch_normalization_101_fusedbatchnormv3_readvariableop_1_resource:@d
Jsequential_60_conv2d_transpose_81_conv2d_transpose_readvariableop_resource: @O
Asequential_60_conv2d_transpose_81_biasadd_readvariableop_resource: K
=sequential_60_batch_normalization_102_readvariableop_resource: M
?sequential_60_batch_normalization_102_readvariableop_1_resource: \
Nsequential_60_batch_normalization_102_fusedbatchnormv3_readvariableop_resource: ^
Psequential_60_batch_normalization_102_fusedbatchnormv3_readvariableop_1_resource: d
Jsequential_60_conv2d_transpose_82_conv2d_transpose_readvariableop_resource: O
Asequential_60_conv2d_transpose_82_biasadd_readvariableop_resource:
identity??Esequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp?Gsequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?4sequential_60/batch_normalization_100/ReadVariableOp?6sequential_60/batch_normalization_100/ReadVariableOp_1?Esequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp?Gsequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?4sequential_60/batch_normalization_101/ReadVariableOp?6sequential_60/batch_normalization_101/ReadVariableOp_1?Esequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp?Gsequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?4sequential_60/batch_normalization_102/ReadVariableOp?6sequential_60/batch_normalization_102/ReadVariableOp_1?Dsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp?Fsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?3sequential_60/batch_normalization_99/ReadVariableOp?5sequential_60/batch_normalization_99/ReadVariableOp_1?/sequential_60/conv2d_112/BiasAdd/ReadVariableOp?.sequential_60/conv2d_112/Conv2D/ReadVariableOp?8sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOp?Asequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOp?8sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOp?Asequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOp?8sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOp?Asequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOp?-sequential_60/dense_40/BiasAdd/ReadVariableOp?,sequential_60/dense_40/MatMul/ReadVariableOp?
,sequential_60/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_60_dense_40_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype02.
,sequential_60/dense_40/MatMul/ReadVariableOp?
sequential_60/dense_40/MatMulMatMuldense_40_input4sequential_60/dense_40/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
sequential_60/dense_40/MatMul?
-sequential_60/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_60_dense_40_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02/
-sequential_60/dense_40/BiasAdd/ReadVariableOp?
sequential_60/dense_40/BiasAddBiasAdd'sequential_60/dense_40/MatMul:product:05sequential_60/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2 
sequential_60/dense_40/BiasAdd?
sequential_60/reshape_20/ShapeShape'sequential_60/dense_40/BiasAdd:output:0*
T0*
_output_shapes
:2 
sequential_60/reshape_20/Shape?
,sequential_60/reshape_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_60/reshape_20/strided_slice/stack?
.sequential_60/reshape_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_60/reshape_20/strided_slice/stack_1?
.sequential_60/reshape_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential_60/reshape_20/strided_slice/stack_2?
&sequential_60/reshape_20/strided_sliceStridedSlice'sequential_60/reshape_20/Shape:output:05sequential_60/reshape_20/strided_slice/stack:output:07sequential_60/reshape_20/strided_slice/stack_1:output:07sequential_60/reshape_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&sequential_60/reshape_20/strided_slice?
(sequential_60/reshape_20/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_60/reshape_20/Reshape/shape/1?
(sequential_60/reshape_20/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_60/reshape_20/Reshape/shape/2?
(sequential_60/reshape_20/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2*
(sequential_60/reshape_20/Reshape/shape/3?
&sequential_60/reshape_20/Reshape/shapePack/sequential_60/reshape_20/strided_slice:output:01sequential_60/reshape_20/Reshape/shape/1:output:01sequential_60/reshape_20/Reshape/shape/2:output:01sequential_60/reshape_20/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&sequential_60/reshape_20/Reshape/shape?
 sequential_60/reshape_20/ReshapeReshape'sequential_60/dense_40/BiasAdd:output:0/sequential_60/reshape_20/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2"
 sequential_60/reshape_20/Reshape?
3sequential_60/batch_normalization_99/ReadVariableOpReadVariableOp<sequential_60_batch_normalization_99_readvariableop_resource*
_output_shapes	
:?*
dtype025
3sequential_60/batch_normalization_99/ReadVariableOp?
5sequential_60/batch_normalization_99/ReadVariableOp_1ReadVariableOp>sequential_60_batch_normalization_99_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5sequential_60/batch_normalization_99/ReadVariableOp_1?
Dsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_60_batch_normalization_99_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp?
Fsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_60_batch_normalization_99_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1?
5sequential_60/batch_normalization_99/FusedBatchNormV3FusedBatchNormV3)sequential_60/reshape_20/Reshape:output:0;sequential_60/batch_normalization_99/ReadVariableOp:value:0=sequential_60/batch_normalization_99/ReadVariableOp_1:value:0Lsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 27
5sequential_60/batch_normalization_99/FusedBatchNormV3?
'sequential_60/conv2d_transpose_80/ShapeShape9sequential_60/batch_normalization_99/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2)
'sequential_60/conv2d_transpose_80/Shape?
5sequential_60/conv2d_transpose_80/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_60/conv2d_transpose_80/strided_slice/stack?
7sequential_60/conv2d_transpose_80/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_60/conv2d_transpose_80/strided_slice/stack_1?
7sequential_60/conv2d_transpose_80/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_60/conv2d_transpose_80/strided_slice/stack_2?
/sequential_60/conv2d_transpose_80/strided_sliceStridedSlice0sequential_60/conv2d_transpose_80/Shape:output:0>sequential_60/conv2d_transpose_80/strided_slice/stack:output:0@sequential_60/conv2d_transpose_80/strided_slice/stack_1:output:0@sequential_60/conv2d_transpose_80/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_60/conv2d_transpose_80/strided_slice?
)sequential_60/conv2d_transpose_80/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_60/conv2d_transpose_80/stack/1?
)sequential_60/conv2d_transpose_80/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_60/conv2d_transpose_80/stack/2?
)sequential_60/conv2d_transpose_80/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_60/conv2d_transpose_80/stack/3?
'sequential_60/conv2d_transpose_80/stackPack8sequential_60/conv2d_transpose_80/strided_slice:output:02sequential_60/conv2d_transpose_80/stack/1:output:02sequential_60/conv2d_transpose_80/stack/2:output:02sequential_60/conv2d_transpose_80/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_60/conv2d_transpose_80/stack?
7sequential_60/conv2d_transpose_80/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_60/conv2d_transpose_80/strided_slice_1/stack?
9sequential_60/conv2d_transpose_80/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_60/conv2d_transpose_80/strided_slice_1/stack_1?
9sequential_60/conv2d_transpose_80/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_60/conv2d_transpose_80/strided_slice_1/stack_2?
1sequential_60/conv2d_transpose_80/strided_slice_1StridedSlice0sequential_60/conv2d_transpose_80/stack:output:0@sequential_60/conv2d_transpose_80/strided_slice_1/stack:output:0Bsequential_60/conv2d_transpose_80/strided_slice_1/stack_1:output:0Bsequential_60/conv2d_transpose_80/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_60/conv2d_transpose_80/strided_slice_1?
Asequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_60_conv2d_transpose_80_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02C
Asequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOp?
2sequential_60/conv2d_transpose_80/conv2d_transposeConv2DBackpropInput0sequential_60/conv2d_transpose_80/stack:output:0Isequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOp:value:09sequential_60/batch_normalization_99/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
24
2sequential_60/conv2d_transpose_80/conv2d_transpose?
8sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOpReadVariableOpAsequential_60_conv2d_transpose_80_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOp?
)sequential_60/conv2d_transpose_80/BiasAddBiasAdd;sequential_60/conv2d_transpose_80/conv2d_transpose:output:0@sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2+
)sequential_60/conv2d_transpose_80/BiasAdd?
&sequential_60/conv2d_transpose_80/SeluSelu2sequential_60/conv2d_transpose_80/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2(
&sequential_60/conv2d_transpose_80/Selu?
4sequential_60/batch_normalization_100/ReadVariableOpReadVariableOp=sequential_60_batch_normalization_100_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_60/batch_normalization_100/ReadVariableOp?
6sequential_60/batch_normalization_100/ReadVariableOp_1ReadVariableOp?sequential_60_batch_normalization_100_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_60/batch_normalization_100/ReadVariableOp_1?
Esequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_60_batch_normalization_100_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp?
Gsequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_60_batch_normalization_100_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1?
6sequential_60/batch_normalization_100/FusedBatchNormV3FusedBatchNormV34sequential_60/conv2d_transpose_80/Selu:activations:0<sequential_60/batch_normalization_100/ReadVariableOp:value:0>sequential_60/batch_normalization_100/ReadVariableOp_1:value:0Msequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp:value:0Osequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 28
6sequential_60/batch_normalization_100/FusedBatchNormV3?
.sequential_60/conv2d_112/Conv2D/ReadVariableOpReadVariableOp7sequential_60_conv2d_112_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.sequential_60/conv2d_112/Conv2D/ReadVariableOp?
sequential_60/conv2d_112/Conv2DConv2D:sequential_60/batch_normalization_100/FusedBatchNormV3:y:06sequential_60/conv2d_112/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2!
sequential_60/conv2d_112/Conv2D?
/sequential_60/conv2d_112/BiasAdd/ReadVariableOpReadVariableOp8sequential_60_conv2d_112_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_60/conv2d_112/BiasAdd/ReadVariableOp?
 sequential_60/conv2d_112/BiasAddBiasAdd(sequential_60/conv2d_112/Conv2D:output:07sequential_60/conv2d_112/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2"
 sequential_60/conv2d_112/BiasAdd?
sequential_60/conv2d_112/SeluSelu)sequential_60/conv2d_112/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
sequential_60/conv2d_112/Selu?
4sequential_60/batch_normalization_101/ReadVariableOpReadVariableOp=sequential_60_batch_normalization_101_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_60/batch_normalization_101/ReadVariableOp?
6sequential_60/batch_normalization_101/ReadVariableOp_1ReadVariableOp?sequential_60_batch_normalization_101_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_60/batch_normalization_101/ReadVariableOp_1?
Esequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_60_batch_normalization_101_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp?
Gsequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_60_batch_normalization_101_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1?
6sequential_60/batch_normalization_101/FusedBatchNormV3FusedBatchNormV3+sequential_60/conv2d_112/Selu:activations:0<sequential_60/batch_normalization_101/ReadVariableOp:value:0>sequential_60/batch_normalization_101/ReadVariableOp_1:value:0Msequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp:value:0Osequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 28
6sequential_60/batch_normalization_101/FusedBatchNormV3?
'sequential_60/conv2d_transpose_81/ShapeShape:sequential_60/batch_normalization_101/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2)
'sequential_60/conv2d_transpose_81/Shape?
5sequential_60/conv2d_transpose_81/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_60/conv2d_transpose_81/strided_slice/stack?
7sequential_60/conv2d_transpose_81/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_60/conv2d_transpose_81/strided_slice/stack_1?
7sequential_60/conv2d_transpose_81/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_60/conv2d_transpose_81/strided_slice/stack_2?
/sequential_60/conv2d_transpose_81/strided_sliceStridedSlice0sequential_60/conv2d_transpose_81/Shape:output:0>sequential_60/conv2d_transpose_81/strided_slice/stack:output:0@sequential_60/conv2d_transpose_81/strided_slice/stack_1:output:0@sequential_60/conv2d_transpose_81/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_60/conv2d_transpose_81/strided_slice?
)sequential_60/conv2d_transpose_81/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_60/conv2d_transpose_81/stack/1?
)sequential_60/conv2d_transpose_81/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_60/conv2d_transpose_81/stack/2?
)sequential_60/conv2d_transpose_81/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_60/conv2d_transpose_81/stack/3?
'sequential_60/conv2d_transpose_81/stackPack8sequential_60/conv2d_transpose_81/strided_slice:output:02sequential_60/conv2d_transpose_81/stack/1:output:02sequential_60/conv2d_transpose_81/stack/2:output:02sequential_60/conv2d_transpose_81/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_60/conv2d_transpose_81/stack?
7sequential_60/conv2d_transpose_81/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_60/conv2d_transpose_81/strided_slice_1/stack?
9sequential_60/conv2d_transpose_81/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_60/conv2d_transpose_81/strided_slice_1/stack_1?
9sequential_60/conv2d_transpose_81/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_60/conv2d_transpose_81/strided_slice_1/stack_2?
1sequential_60/conv2d_transpose_81/strided_slice_1StridedSlice0sequential_60/conv2d_transpose_81/stack:output:0@sequential_60/conv2d_transpose_81/strided_slice_1/stack:output:0Bsequential_60/conv2d_transpose_81/strided_slice_1/stack_1:output:0Bsequential_60/conv2d_transpose_81/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_60/conv2d_transpose_81/strided_slice_1?
Asequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_60_conv2d_transpose_81_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02C
Asequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOp?
2sequential_60/conv2d_transpose_81/conv2d_transposeConv2DBackpropInput0sequential_60/conv2d_transpose_81/stack:output:0Isequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOp:value:0:sequential_60/batch_normalization_101/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
24
2sequential_60/conv2d_transpose_81/conv2d_transpose?
8sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOpReadVariableOpAsequential_60_conv2d_transpose_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOp?
)sequential_60/conv2d_transpose_81/BiasAddBiasAdd;sequential_60/conv2d_transpose_81/conv2d_transpose:output:0@sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2+
)sequential_60/conv2d_transpose_81/BiasAdd?
&sequential_60/conv2d_transpose_81/SeluSelu2sequential_60/conv2d_transpose_81/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2(
&sequential_60/conv2d_transpose_81/Selu?
4sequential_60/batch_normalization_102/ReadVariableOpReadVariableOp=sequential_60_batch_normalization_102_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_60/batch_normalization_102/ReadVariableOp?
6sequential_60/batch_normalization_102/ReadVariableOp_1ReadVariableOp?sequential_60_batch_normalization_102_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential_60/batch_normalization_102/ReadVariableOp_1?
Esequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_60_batch_normalization_102_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Esequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp?
Gsequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_60_batch_normalization_102_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gsequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1?
6sequential_60/batch_normalization_102/FusedBatchNormV3FusedBatchNormV34sequential_60/conv2d_transpose_81/Selu:activations:0<sequential_60/batch_normalization_102/ReadVariableOp:value:0>sequential_60/batch_normalization_102/ReadVariableOp_1:value:0Msequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp:value:0Osequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 28
6sequential_60/batch_normalization_102/FusedBatchNormV3?
'sequential_60/conv2d_transpose_82/ShapeShape:sequential_60/batch_normalization_102/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2)
'sequential_60/conv2d_transpose_82/Shape?
5sequential_60/conv2d_transpose_82/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_60/conv2d_transpose_82/strided_slice/stack?
7sequential_60/conv2d_transpose_82/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_60/conv2d_transpose_82/strided_slice/stack_1?
7sequential_60/conv2d_transpose_82/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_60/conv2d_transpose_82/strided_slice/stack_2?
/sequential_60/conv2d_transpose_82/strided_sliceStridedSlice0sequential_60/conv2d_transpose_82/Shape:output:0>sequential_60/conv2d_transpose_82/strided_slice/stack:output:0@sequential_60/conv2d_transpose_82/strided_slice/stack_1:output:0@sequential_60/conv2d_transpose_82/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_60/conv2d_transpose_82/strided_slice?
)sequential_60/conv2d_transpose_82/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_60/conv2d_transpose_82/stack/1?
)sequential_60/conv2d_transpose_82/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_60/conv2d_transpose_82/stack/2?
)sequential_60/conv2d_transpose_82/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_60/conv2d_transpose_82/stack/3?
'sequential_60/conv2d_transpose_82/stackPack8sequential_60/conv2d_transpose_82/strided_slice:output:02sequential_60/conv2d_transpose_82/stack/1:output:02sequential_60/conv2d_transpose_82/stack/2:output:02sequential_60/conv2d_transpose_82/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_60/conv2d_transpose_82/stack?
7sequential_60/conv2d_transpose_82/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_60/conv2d_transpose_82/strided_slice_1/stack?
9sequential_60/conv2d_transpose_82/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_60/conv2d_transpose_82/strided_slice_1/stack_1?
9sequential_60/conv2d_transpose_82/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_60/conv2d_transpose_82/strided_slice_1/stack_2?
1sequential_60/conv2d_transpose_82/strided_slice_1StridedSlice0sequential_60/conv2d_transpose_82/stack:output:0@sequential_60/conv2d_transpose_82/strided_slice_1/stack:output:0Bsequential_60/conv2d_transpose_82/strided_slice_1/stack_1:output:0Bsequential_60/conv2d_transpose_82/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_60/conv2d_transpose_82/strided_slice_1?
Asequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_60_conv2d_transpose_82_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02C
Asequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOp?
2sequential_60/conv2d_transpose_82/conv2d_transposeConv2DBackpropInput0sequential_60/conv2d_transpose_82/stack:output:0Isequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOp:value:0:sequential_60/batch_normalization_102/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
24
2sequential_60/conv2d_transpose_82/conv2d_transpose?
8sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOpReadVariableOpAsequential_60_conv2d_transpose_82_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOp?
)sequential_60/conv2d_transpose_82/BiasAddBiasAdd;sequential_60/conv2d_transpose_82/conv2d_transpose:output:0@sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)sequential_60/conv2d_transpose_82/BiasAdd?
&sequential_60/conv2d_transpose_82/TanhTanh2sequential_60/conv2d_transpose_82/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2(
&sequential_60/conv2d_transpose_82/Tanh?
IdentityIdentity*sequential_60/conv2d_transpose_82/Tanh:y:0F^sequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOpH^sequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_15^sequential_60/batch_normalization_100/ReadVariableOp7^sequential_60/batch_normalization_100/ReadVariableOp_1F^sequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOpH^sequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_15^sequential_60/batch_normalization_101/ReadVariableOp7^sequential_60/batch_normalization_101/ReadVariableOp_1F^sequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOpH^sequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_15^sequential_60/batch_normalization_102/ReadVariableOp7^sequential_60/batch_normalization_102/ReadVariableOp_1E^sequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOpG^sequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_14^sequential_60/batch_normalization_99/ReadVariableOp6^sequential_60/batch_normalization_99/ReadVariableOp_10^sequential_60/conv2d_112/BiasAdd/ReadVariableOp/^sequential_60/conv2d_112/Conv2D/ReadVariableOp9^sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOpB^sequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOp9^sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOpB^sequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOp9^sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOpB^sequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOp.^sequential_60/dense_40/BiasAdd/ReadVariableOp-^sequential_60/dense_40/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Esequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOpEsequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp2?
Gsequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_1Gsequential_60/batch_normalization_100/FusedBatchNormV3/ReadVariableOp_12l
4sequential_60/batch_normalization_100/ReadVariableOp4sequential_60/batch_normalization_100/ReadVariableOp2p
6sequential_60/batch_normalization_100/ReadVariableOp_16sequential_60/batch_normalization_100/ReadVariableOp_12?
Esequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOpEsequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp2?
Gsequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_1Gsequential_60/batch_normalization_101/FusedBatchNormV3/ReadVariableOp_12l
4sequential_60/batch_normalization_101/ReadVariableOp4sequential_60/batch_normalization_101/ReadVariableOp2p
6sequential_60/batch_normalization_101/ReadVariableOp_16sequential_60/batch_normalization_101/ReadVariableOp_12?
Esequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOpEsequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp2?
Gsequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_1Gsequential_60/batch_normalization_102/FusedBatchNormV3/ReadVariableOp_12l
4sequential_60/batch_normalization_102/ReadVariableOp4sequential_60/batch_normalization_102/ReadVariableOp2p
6sequential_60/batch_normalization_102/ReadVariableOp_16sequential_60/batch_normalization_102/ReadVariableOp_12?
Dsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOpDsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp2?
Fsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_1Fsequential_60/batch_normalization_99/FusedBatchNormV3/ReadVariableOp_12j
3sequential_60/batch_normalization_99/ReadVariableOp3sequential_60/batch_normalization_99/ReadVariableOp2n
5sequential_60/batch_normalization_99/ReadVariableOp_15sequential_60/batch_normalization_99/ReadVariableOp_12b
/sequential_60/conv2d_112/BiasAdd/ReadVariableOp/sequential_60/conv2d_112/BiasAdd/ReadVariableOp2`
.sequential_60/conv2d_112/Conv2D/ReadVariableOp.sequential_60/conv2d_112/Conv2D/ReadVariableOp2t
8sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOp8sequential_60/conv2d_transpose_80/BiasAdd/ReadVariableOp2?
Asequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOpAsequential_60/conv2d_transpose_80/conv2d_transpose/ReadVariableOp2t
8sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOp8sequential_60/conv2d_transpose_81/BiasAdd/ReadVariableOp2?
Asequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOpAsequential_60/conv2d_transpose_81/conv2d_transpose/ReadVariableOp2t
8sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOp8sequential_60/conv2d_transpose_82/BiasAdd/ReadVariableOp2?
Asequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOpAsequential_60/conv2d_transpose_82/conv2d_transpose/ReadVariableOp2^
-sequential_60/dense_40/BiasAdd/ReadVariableOp-sequential_60/dense_40/BiasAdd/ReadVariableOp2\
,sequential_60/dense_40/MatMul/ReadVariableOp,sequential_60/dense_40/MatMul/ReadVariableOp:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_40_input
?
?
0__inference_sequential_60_layer_call_fn_19454815

inputs
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?$
	unknown_5:@?
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17: @

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_60_layer_call_and_return_conditional_losses_194542052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?@
?
!__inference__traced_save_19455627
file_prefix.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop;
7savev2_batch_normalization_99_gamma_read_readvariableop:
6savev2_batch_normalization_99_beta_read_readvariableopA
=savev2_batch_normalization_99_moving_mean_read_readvariableopE
Asavev2_batch_normalization_99_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_80_kernel_read_readvariableop7
3savev2_conv2d_transpose_80_bias_read_readvariableop<
8savev2_batch_normalization_100_gamma_read_readvariableop;
7savev2_batch_normalization_100_beta_read_readvariableopB
>savev2_batch_normalization_100_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_100_moving_variance_read_readvariableop0
,savev2_conv2d_112_kernel_read_readvariableop.
*savev2_conv2d_112_bias_read_readvariableop<
8savev2_batch_normalization_101_gamma_read_readvariableop;
7savev2_batch_normalization_101_beta_read_readvariableopB
>savev2_batch_normalization_101_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_101_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_81_kernel_read_readvariableop7
3savev2_conv2d_transpose_81_bias_read_readvariableop<
8savev2_batch_normalization_102_gamma_read_readvariableop;
7savev2_batch_normalization_102_beta_read_readvariableopB
>savev2_batch_normalization_102_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_102_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_82_kernel_read_readvariableop7
3savev2_conv2d_transpose_82_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop7savev2_batch_normalization_99_gamma_read_readvariableop6savev2_batch_normalization_99_beta_read_readvariableop=savev2_batch_normalization_99_moving_mean_read_readvariableopAsavev2_batch_normalization_99_moving_variance_read_readvariableop5savev2_conv2d_transpose_80_kernel_read_readvariableop3savev2_conv2d_transpose_80_bias_read_readvariableop8savev2_batch_normalization_100_gamma_read_readvariableop7savev2_batch_normalization_100_beta_read_readvariableop>savev2_batch_normalization_100_moving_mean_read_readvariableopBsavev2_batch_normalization_100_moving_variance_read_readvariableop,savev2_conv2d_112_kernel_read_readvariableop*savev2_conv2d_112_bias_read_readvariableop8savev2_batch_normalization_101_gamma_read_readvariableop7savev2_batch_normalization_101_beta_read_readvariableop>savev2_batch_normalization_101_moving_mean_read_readvariableopBsavev2_batch_normalization_101_moving_variance_read_readvariableop5savev2_conv2d_transpose_81_kernel_read_readvariableop3savev2_conv2d_transpose_81_bias_read_readvariableop8savev2_batch_normalization_102_gamma_read_readvariableop7savev2_batch_normalization_102_beta_read_readvariableop>savev2_batch_normalization_102_moving_mean_read_readvariableopBsavev2_batch_normalization_102_moving_variance_read_readvariableop5savev2_conv2d_transpose_82_kernel_read_readvariableop3savev2_conv2d_transpose_82_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :???:??:?:?:?:?:@?:@:@:@:@:@:@@:@:@:@:@:@: @: : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:'#
!
_output_shapes
:???:"

_output_shapes

:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?
?
0__inference_sequential_60_layer_call_fn_19454567
dense_40_input
unknown:???
	unknown_0:
??
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?$
	unknown_5:@?
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17: @

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: $

unknown_23: 

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_60_layer_call_and_return_conditional_losses_194544552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_40_input
?
?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19455526

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19453677

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_reshape_20_layer_call_and_return_conditional_losses_19455196

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_dense_40_layer_call_fn_19455167

inputs
unknown:???
	unknown_0:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_40_layer_call_and_return_conditional_losses_194540962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_101_layer_call_fn_19455415

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_194537592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
6__inference_conv2d_transpose_82_layer_call_fn_19454079

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_194540692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
:__inference_batch_normalization_102_layer_call_fn_19455477

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_194539302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
F__inference_dense_40_layer_call_and_return_conditional_losses_19455177

inputs3
matmul_readvariableop_resource:???/
biasadd_readvariableop_resource:
??
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:??*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_99_layer_call_fn_19455209

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_194534622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19455508

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?E
?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454633
dense_40_input&
dense_40_19454570:???!
dense_40_19454572:
??.
batch_normalization_99_19454576:	?.
batch_normalization_99_19454578:	?.
batch_normalization_99_19454580:	?.
batch_normalization_99_19454582:	?7
conv2d_transpose_80_19454585:@?*
conv2d_transpose_80_19454587:@.
 batch_normalization_100_19454590:@.
 batch_normalization_100_19454592:@.
 batch_normalization_100_19454594:@.
 batch_normalization_100_19454596:@-
conv2d_112_19454599:@@!
conv2d_112_19454601:@.
 batch_normalization_101_19454604:@.
 batch_normalization_101_19454606:@.
 batch_normalization_101_19454608:@.
 batch_normalization_101_19454610:@6
conv2d_transpose_81_19454613: @*
conv2d_transpose_81_19454615: .
 batch_normalization_102_19454618: .
 batch_normalization_102_19454620: .
 batch_normalization_102_19454622: .
 batch_normalization_102_19454624: 6
conv2d_transpose_82_19454627: *
conv2d_transpose_82_19454629:
identity??/batch_normalization_100/StatefulPartitionedCall?/batch_normalization_101/StatefulPartitionedCall?/batch_normalization_102/StatefulPartitionedCall?.batch_normalization_99/StatefulPartitionedCall?"conv2d_112/StatefulPartitionedCall?+conv2d_transpose_80/StatefulPartitionedCall?+conv2d_transpose_81/StatefulPartitionedCall?+conv2d_transpose_82/StatefulPartitionedCall? dense_40/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCalldense_40_inputdense_40_19454570dense_40_19454572*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_40_layer_call_and_return_conditional_losses_194540962"
 dense_40/StatefulPartitionedCall?
reshape_20/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reshape_20_layer_call_and_return_conditional_losses_194541162
reshape_20/PartitionedCall?
.batch_normalization_99/StatefulPartitionedCallStatefulPartitionedCall#reshape_20/PartitionedCall:output:0batch_normalization_99_19454576batch_normalization_99_19454578batch_normalization_99_19454580batch_normalization_99_19454582*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_1945413520
.batch_normalization_99/StatefulPartitionedCall?
+conv2d_transpose_80/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_99/StatefulPartitionedCall:output:0conv2d_transpose_80_19454585conv2d_transpose_80_19454587*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_194536012-
+conv2d_transpose_80/StatefulPartitionedCall?
/batch_normalization_100/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_80/StatefulPartitionedCall:output:0 batch_normalization_100_19454590 batch_normalization_100_19454592 batch_normalization_100_19454594 batch_normalization_100_19454596*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_1945363321
/batch_normalization_100/StatefulPartitionedCall?
"conv2d_112/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_100/StatefulPartitionedCall:output:0conv2d_112_19454599conv2d_112_19454601*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_112_layer_call_and_return_conditional_losses_194541702$
"conv2d_112/StatefulPartitionedCall?
/batch_normalization_101/StatefulPartitionedCallStatefulPartitionedCall+conv2d_112/StatefulPartitionedCall:output:0 batch_normalization_101_19454604 batch_normalization_101_19454606 batch_normalization_101_19454608 batch_normalization_101_19454610*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_1945375921
/batch_normalization_101/StatefulPartitionedCall?
+conv2d_transpose_81/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_101/StatefulPartitionedCall:output:0conv2d_transpose_81_19454613conv2d_transpose_81_19454615*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_194538982-
+conv2d_transpose_81/StatefulPartitionedCall?
/batch_normalization_102/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_81/StatefulPartitionedCall:output:0 batch_normalization_102_19454618 batch_normalization_102_19454620 batch_normalization_102_19454622 batch_normalization_102_19454624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_1945393021
/batch_normalization_102/StatefulPartitionedCall?
+conv2d_transpose_82/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_102/StatefulPartitionedCall:output:0conv2d_transpose_82_19454627conv2d_transpose_82_19454629*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_194540692-
+conv2d_transpose_82/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_82/StatefulPartitionedCall:output:00^batch_normalization_100/StatefulPartitionedCall0^batch_normalization_101/StatefulPartitionedCall0^batch_normalization_102/StatefulPartitionedCall/^batch_normalization_99/StatefulPartitionedCall#^conv2d_112/StatefulPartitionedCall,^conv2d_transpose_80/StatefulPartitionedCall,^conv2d_transpose_81/StatefulPartitionedCall,^conv2d_transpose_82/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:??????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_100/StatefulPartitionedCall/batch_normalization_100/StatefulPartitionedCall2b
/batch_normalization_101/StatefulPartitionedCall/batch_normalization_101/StatefulPartitionedCall2b
/batch_normalization_102/StatefulPartitionedCall/batch_normalization_102/StatefulPartitionedCall2`
.batch_normalization_99/StatefulPartitionedCall.batch_normalization_99/StatefulPartitionedCall2H
"conv2d_112/StatefulPartitionedCall"conv2d_112/StatefulPartitionedCall2Z
+conv2d_transpose_80/StatefulPartitionedCall+conv2d_transpose_80/StatefulPartitionedCall2Z
+conv2d_transpose_81/StatefulPartitionedCall+conv2d_transpose_81/StatefulPartitionedCall2Z
+conv2d_transpose_82/StatefulPartitionedCall+conv2d_transpose_82/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_40_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
J
dense_40_input8
 serving_default_dense_40_input:0??????????Q
conv2d_transpose_82:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?n
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?j
_tf_keras_sequential?j{"name": "sequential_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_40_input"}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_20", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16, 128]}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_99", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_80", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_100", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_112", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_101", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_81", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_102", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_82", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300]}, "float32", "dense_40_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_40_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "reshape_20", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16, 128]}}, "shared_object_id": 4}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_99", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_80", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 12}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_100", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_112", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_101", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 25}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_81", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 28}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_102", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 30}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 33}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_82", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 36}]}}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "reshape_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_20", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16, 128]}}, "shared_object_id": 4}
?

axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
 	variables
!trainable_variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_99", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_99", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
?

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_transpose_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_80", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
?

)axis
	*gamma
+beta
,moving_mean
-moving_variance
.regularization_losses
/	variables
0trainable_variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_100", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_100", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?


2kernel
3bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_112", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_112", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?

8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_101", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_101", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_transpose_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_81", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?

Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_102", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_102", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 30}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
?

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_transpose_82", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_82", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
 "
trackable_list_wrapper
?
0
1
2
3
4
5
#6
$7
*8
+9
,10
-11
212
313
914
:15
;16
<17
A18
B19
H20
I21
J22
K23
P24
Q25"
trackable_list_wrapper
?
0
1
2
3
#4
$5
*6
+7
28
39
910
:11
A12
B13
H14
I15
P16
Q17"
trackable_list_wrapper
?
Vlayer_metrics
regularization_losses

Wlayers
	variables
trainable_variables
Xlayer_regularization_losses
Ymetrics
Znon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
$:"???2dense_40/kernel
:??2dense_40/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
[layer_metrics
regularization_losses
\layer_regularization_losses
	variables
trainable_variables

]layers
^metrics
_non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
`layer_metrics
regularization_losses
alayer_regularization_losses
	variables
trainable_variables

blayers
cmetrics
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_99/gamma
*:(?2batch_normalization_99/beta
3:1? (2"batch_normalization_99/moving_mean
7:5? (2&batch_normalization_99/moving_variance
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
elayer_metrics
regularization_losses
flayer_regularization_losses
 	variables
!trainable_variables

glayers
hmetrics
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3@?2conv2d_transpose_80/kernel
&:$@2conv2d_transpose_80/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
jlayer_metrics
%regularization_losses
klayer_regularization_losses
&	variables
'trainable_variables

llayers
mmetrics
nnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2batch_normalization_100/gamma
*:(@2batch_normalization_100/beta
3:1@ (2#batch_normalization_100/moving_mean
7:5@ (2'batch_normalization_100/moving_variance
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
olayer_metrics
.regularization_losses
player_regularization_losses
/	variables
0trainable_variables

qlayers
rmetrics
snon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@@2conv2d_112/kernel
:@2conv2d_112/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
tlayer_metrics
4regularization_losses
ulayer_regularization_losses
5	variables
6trainable_variables

vlayers
wmetrics
xnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2batch_normalization_101/gamma
*:(@2batch_normalization_101/beta
3:1@ (2#batch_normalization_101/moving_mean
7:5@ (2'batch_normalization_101/moving_variance
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
ylayer_metrics
=regularization_losses
zlayer_regularization_losses
>	variables
?trainable_variables

{layers
|metrics
}non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2 @2conv2d_transpose_81/kernel
&:$ 2conv2d_transpose_81/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
~layer_metrics
Cregularization_losses
layer_regularization_losses
D	variables
Etrainable_variables
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_102/gamma
*:( 2batch_normalization_102/beta
3:1  (2#batch_normalization_102/moving_mean
7:5  (2'batch_normalization_102/moving_variance
 "
trackable_list_wrapper
<
H0
I1
J2
K3"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?layer_metrics
Lregularization_losses
 ?layer_regularization_losses
M	variables
Ntrainable_variables
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2 2conv2d_transpose_82/kernel
&:$2conv2d_transpose_82/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?layer_metrics
Rregularization_losses
 ?layer_regularization_losses
S	variables
Ttrainable_variables
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
,2
-3
;4
<5
J6
K7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
#__inference__wrapped_model_19453440?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
dense_40_input??????????
?2?
0__inference_sequential_60_layer_call_fn_19454260
0__inference_sequential_60_layer_call_fn_19454815
0__inference_sequential_60_layer_call_fn_19454872
0__inference_sequential_60_layer_call_fn_19454567?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19455015
K__inference_sequential_60_layer_call_and_return_conditional_losses_19455158
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454633
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454699?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dense_40_layer_call_fn_19455167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_40_layer_call_and_return_conditional_losses_19455177?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_reshape_20_layer_call_fn_19455182?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_reshape_20_layer_call_and_return_conditional_losses_19455196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_99_layer_call_fn_19455209
9__inference_batch_normalization_99_layer_call_fn_19455222
9__inference_batch_normalization_99_layer_call_fn_19455235
9__inference_batch_normalization_99_layer_call_fn_19455248?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455266
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455284
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455302
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455320?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_conv2d_transpose_80_layer_call_fn_19453611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_19453601?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
:__inference_batch_normalization_100_layer_call_fn_19455333
:__inference_batch_normalization_100_layer_call_fn_19455346?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19455364
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19455382?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_conv2d_112_layer_call_fn_19455391?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_112_layer_call_and_return_conditional_losses_19455402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_batch_normalization_101_layer_call_fn_19455415
:__inference_batch_normalization_101_layer_call_fn_19455428?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19455446
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19455464?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_conv2d_transpose_81_layer_call_fn_19453908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_19453898?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
:__inference_batch_normalization_102_layer_call_fn_19455477
:__inference_batch_normalization_102_layer_call_fn_19455490?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19455508
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19455526?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_conv2d_transpose_82_layer_call_fn_19454079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_19454069?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?B?
&__inference_signature_wrapper_19454758dense_40_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_19453440?#$*+,-239:;<ABHIJKPQ8?5
.?+
)?&
dense_40_input??????????
? "S?P
N
conv2d_transpose_827?4
conv2d_transpose_82????????????
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19455364?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
U__inference_batch_normalization_100_layer_call_and_return_conditional_losses_19455382?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
:__inference_batch_normalization_100_layer_call_fn_19455333?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
:__inference_batch_normalization_100_layer_call_fn_19455346?*+,-M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19455446?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
U__inference_batch_normalization_101_layer_call_and_return_conditional_losses_19455464?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
:__inference_batch_normalization_101_layer_call_fn_19455415?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
:__inference_batch_normalization_101_layer_call_fn_19455428?9:;<M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19455508?HIJKM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
U__inference_batch_normalization_102_layer_call_and_return_conditional_losses_19455526?HIJKM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
:__inference_batch_normalization_102_layer_call_fn_19455477?HIJKM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
:__inference_batch_normalization_102_layer_call_fn_19455490?HIJKM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455266?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455284?N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455302t<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_99_layer_call_and_return_conditional_losses_19455320t<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_99_layer_call_fn_19455209?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_99_layer_call_fn_19455222?N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_99_layer_call_fn_19455235g<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_99_layer_call_fn_19455248g<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_conv2d_112_layer_call_and_return_conditional_losses_19455402?23I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
-__inference_conv2d_112_layer_call_fn_19455391?23I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
Q__inference_conv2d_transpose_80_layer_call_and_return_conditional_losses_19453601?#$J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
6__inference_conv2d_transpose_80_layer_call_fn_19453611?#$J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
Q__inference_conv2d_transpose_81_layer_call_and_return_conditional_losses_19453898?ABI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_conv2d_transpose_81_layer_call_fn_19453908?ABI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
Q__inference_conv2d_transpose_82_layer_call_and_return_conditional_losses_19454069?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
6__inference_conv2d_transpose_82_layer_call_fn_19454079?PQI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
F__inference_dense_40_layer_call_and_return_conditional_losses_19455177_0?-
&?#
!?
inputs??????????
? "'?$
?
0???????????
? ?
+__inference_dense_40_layer_call_fn_19455167R0?-
&?#
!?
inputs??????????
? "?????????????
H__inference_reshape_20_layer_call_and_return_conditional_losses_19455196c1?.
'?$
"?
inputs???????????
? ".?+
$?!
0??????????
? ?
-__inference_reshape_20_layer_call_fn_19455182V1?.
'?$
"?
inputs???????????
? "!????????????
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454633?#$*+,-239:;<ABHIJKPQ@?=
6?3
)?&
dense_40_input??????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19454699?#$*+,-239:;<ABHIJKPQ@?=
6?3
)?&
dense_40_input??????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19455015?#$*+,-239:;<ABHIJKPQ8?5
.?+
!?
inputs??????????
p 

 
? "/?,
%?"
0???????????
? ?
K__inference_sequential_60_layer_call_and_return_conditional_losses_19455158?#$*+,-239:;<ABHIJKPQ8?5
.?+
!?
inputs??????????
p

 
? "/?,
%?"
0???????????
? ?
0__inference_sequential_60_layer_call_fn_19454260?#$*+,-239:;<ABHIJKPQ@?=
6?3
)?&
dense_40_input??????????
p 

 
? "2?/+????????????????????????????
0__inference_sequential_60_layer_call_fn_19454567?#$*+,-239:;<ABHIJKPQ@?=
6?3
)?&
dense_40_input??????????
p

 
? "2?/+????????????????????????????
0__inference_sequential_60_layer_call_fn_19454815?#$*+,-239:;<ABHIJKPQ8?5
.?+
!?
inputs??????????
p 

 
? "2?/+????????????????????????????
0__inference_sequential_60_layer_call_fn_19454872?#$*+,-239:;<ABHIJKPQ8?5
.?+
!?
inputs??????????
p

 
? "2?/+????????????????????????????
&__inference_signature_wrapper_19454758?#$*+,-239:;<ABHIJKPQJ?G
? 
@?=
;
dense_40_input)?&
dense_40_input??????????"S?P
N
conv2d_transpose_827?4
conv2d_transpose_82???????????