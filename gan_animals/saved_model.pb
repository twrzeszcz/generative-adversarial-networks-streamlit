??
??
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
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d??* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
d??*
dtype0
t
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_namedense_14/bias
m
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes

:??*
dtype0
?
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_26/gamma
?
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_26/beta
?
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_26/moving_mean
?
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_26/moving_variance
?
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameconv2d_transpose_21/kernel
?
.conv2d_transpose_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_21/kernel*'
_output_shapes
:@?*
dtype0
?
conv2d_transpose_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_21/bias
?
,conv2d_transpose_21/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_21/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_22/kernel
?
.conv2d_transpose_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_22/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_22/bias
?
,conv2d_transpose_22/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_22/bias*
_output_shapes
: *
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
: *
dtype0
?
conv2d_transpose_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_23/kernel
?
.conv2d_transpose_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_23/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_23/bias
?
,conv2d_transpose_23/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_23/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?.
value?-B?- B?-
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
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?
0
1
2
3
4
5
!6
"7
(8
)9
*10
+11
012
113
714
815
916
:17
?18
@19
f
0
1
2
3
!4
"5
(6
)7
08
19
710
811
?12
@13
 
?
Emetrics
		variables

Flayers
Gnon_trainable_variables
Hlayer_metrics

trainable_variables
Ilayer_regularization_losses
regularization_losses
 
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Jmetrics

Klayers
Lnon_trainable_variables
	variables
Mlayer_metrics
trainable_variables
Nlayer_regularization_losses
regularization_losses
 
 
 
?
Ometrics

Players
Qnon_trainable_variables
	variables
Rlayer_metrics
trainable_variables
Slayer_regularization_losses
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_26/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_26/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_26/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_26/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
?
Tmetrics

Ulayers
Vnon_trainable_variables
	variables
Wlayer_metrics
trainable_variables
Xlayer_regularization_losses
regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_21/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_21/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
Ymetrics

Zlayers
[non_trainable_variables
#	variables
\layer_metrics
$trainable_variables
]layer_regularization_losses
%regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_27/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_27/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_27/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_27/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
*2
+3

(0
)1
 
?
^metrics

_layers
`non_trainable_variables
,	variables
alayer_metrics
-trainable_variables
blayer_regularization_losses
.regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_22/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_22/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?
cmetrics

dlayers
enon_trainable_variables
2	variables
flayer_metrics
3trainable_variables
glayer_regularization_losses
4regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_28/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_28/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_28/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_28/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

70
81
92
:3

70
81
 
?
hmetrics

ilayers
jnon_trainable_variables
;	variables
klayer_metrics
<trainable_variables
llayer_regularization_losses
=regularization_losses
fd
VARIABLE_VALUEconv2d_transpose_23/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_23/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
mmetrics

nlayers
onon_trainable_variables
A	variables
player_metrics
Btrainable_variables
qlayer_regularization_losses
Cregularization_losses
 
8
0
1
2
3
4
5
6
7
*
0
1
*2
+3
94
:5
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
0
1
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
*0
+1
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
90
:1
 
 
 
 
 
 
 
?
serving_default_dense_14_inputPlaceholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_14_inputdense_14/kerneldense_14/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_transpose_21/kernelconv2d_transpose_21/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_transpose_22/kernelconv2d_transpose_22/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_23/kernelconv2d_transpose_23/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_10975254
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp.conv2d_transpose_21/kernel/Read/ReadVariableOp,conv2d_transpose_21/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp.conv2d_transpose_22/kernel/Read/ReadVariableOp,conv2d_transpose_22/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp.conv2d_transpose_23/kernel/Read/ReadVariableOp,conv2d_transpose_23/bias/Read/ReadVariableOpConst*!
Tin
2*
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
!__inference__traced_save_10975957
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biasbatch_normalization_26/gammabatch_normalization_26/beta"batch_normalization_26/moving_mean&batch_normalization_26/moving_varianceconv2d_transpose_21/kernelconv2d_transpose_21/biasbatch_normalization_27/gammabatch_normalization_27/beta"batch_normalization_27/moving_mean&batch_normalization_27/moving_varianceconv2d_transpose_22/kernelconv2d_transpose_22/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_transpose_23/kernelconv2d_transpose_23/bias* 
Tin
2*
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
$__inference__traced_restore_10976027??
?
?
9__inference_batch_normalization_26_layer_call_fn_10975665

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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_109747692
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
?%
?
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_10974703

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
9__inference_batch_normalization_27_layer_call_fn_10975776

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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_109744372
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
9__inference_batch_normalization_28_layer_call_fn_10975825

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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_109745642
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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10974222

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
?
?
9__inference_batch_normalization_26_layer_call_fn_10975639

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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_109742222
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
6__inference_conv2d_transpose_23_layer_call_fn_10974713

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
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_109747032
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
6__inference_conv2d_transpose_21_layer_call_fn_10974371

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
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_109743612
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
?
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975732

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
?5
?

!__inference__traced_save_10975957
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop;
7savev2_batch_normalization_26_gamma_read_readvariableop:
6savev2_batch_normalization_26_beta_read_readvariableopA
=savev2_batch_normalization_26_moving_mean_read_readvariableopE
Asavev2_batch_normalization_26_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_21_kernel_read_readvariableop7
3savev2_conv2d_transpose_21_bias_read_readvariableop;
7savev2_batch_normalization_27_gamma_read_readvariableop:
6savev2_batch_normalization_27_beta_read_readvariableopA
=savev2_batch_normalization_27_moving_mean_read_readvariableopE
Asavev2_batch_normalization_27_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_22_kernel_read_readvariableop7
3savev2_conv2d_transpose_22_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_23_kernel_read_readvariableop7
3savev2_conv2d_transpose_23_bias_read_readvariableop
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
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop7savev2_batch_normalization_26_gamma_read_readvariableop6savev2_batch_normalization_26_beta_read_readvariableop=savev2_batch_normalization_26_moving_mean_read_readvariableopAsavev2_batch_normalization_26_moving_variance_read_readvariableop5savev2_conv2d_transpose_21_kernel_read_readvariableop3savev2_conv2d_transpose_21_bias_read_readvariableop7savev2_batch_normalization_27_gamma_read_readvariableop6savev2_batch_normalization_27_beta_read_readvariableop=savev2_batch_normalization_27_moving_mean_read_readvariableopAsavev2_batch_normalization_27_moving_variance_read_readvariableop5savev2_conv2d_transpose_22_kernel_read_readvariableop3savev2_conv2d_transpose_22_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop5savev2_conv2d_transpose_23_kernel_read_readvariableop3savev2_conv2d_transpose_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
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
?: :
d??:??:?:?:?:?:@?:@:@:@:@:@: @: : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
d??:"

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
: @: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?	
?
F__inference_dense_14_layer_call_and_return_conditional_losses_10975607

inputs2
matmul_readvariableop_resource:
d??/
biasadd_readvariableop_resource:
??
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d??*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975588

inputs;
'dense_14_matmul_readvariableop_resource:
d??8
(dense_14_biasadd_readvariableop_resource:
??=
.batch_normalization_26_readvariableop_resource:	??
0batch_normalization_26_readvariableop_1_resource:	?N
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_21_conv2d_transpose_readvariableop_resource:@?A
3conv2d_transpose_21_biasadd_readvariableop_resource:@<
.batch_normalization_27_readvariableop_resource:@>
0batch_normalization_27_readvariableop_1_resource:@M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_22_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_22_biasadd_readvariableop_resource: <
.batch_normalization_28_readvariableop_resource: >
0batch_normalization_28_readvariableop_1_resource: M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_23_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_23_biasadd_readvariableop_resource:
identity??%batch_normalization_26/AssignNewValue?'batch_normalization_26/AssignNewValue_1?6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?%batch_normalization_27/AssignNewValue?'batch_normalization_27/AssignNewValue_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?*conv2d_transpose_21/BiasAdd/ReadVariableOp?3conv2d_transpose_21/conv2d_transpose/ReadVariableOp?*conv2d_transpose_22/BiasAdd/ReadVariableOp?3conv2d_transpose_22/conv2d_transpose/ReadVariableOp?*conv2d_transpose_23/BiasAdd/ReadVariableOp?3conv2d_transpose_23/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/BiasAddk
reshape_7/ShapeShapedense_14/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_7/Shape?
reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_7/strided_slice/stack?
reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_7/strided_slice/stack_1?
reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_7/strided_slice/stack_2?
reshape_7/strided_sliceStridedSlicereshape_7/Shape:output:0&reshape_7/strided_slice/stack:output:0(reshape_7/strided_slice/stack_1:output:0(reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_7/strided_slicex
reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_7/Reshape/shape/1x
reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_7/Reshape/shape/2y
reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_7/Reshape/shape/3?
reshape_7/Reshape/shapePack reshape_7/strided_slice:output:0"reshape_7/Reshape/shape/1:output:0"reshape_7/Reshape/shape/2:output:0"reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_7/Reshape/shape?
reshape_7/ReshapeReshapedense_14/BiasAdd:output:0 reshape_7/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_7/Reshape?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3reshape_7/Reshape:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_26/FusedBatchNormV3?
%batch_normalization_26/AssignNewValueAssignVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource4batch_normalization_26/FusedBatchNormV3:batch_mean:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_26/AssignNewValue?
'batch_normalization_26/AssignNewValue_1AssignVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_26/FusedBatchNormV3:batch_variance:09^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_26/AssignNewValue_1?
conv2d_transpose_21/ShapeShape+batch_normalization_26/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_21/Shape?
'conv2d_transpose_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_21/strided_slice/stack?
)conv2d_transpose_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_21/strided_slice/stack_1?
)conv2d_transpose_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_21/strided_slice/stack_2?
!conv2d_transpose_21/strided_sliceStridedSlice"conv2d_transpose_21/Shape:output:00conv2d_transpose_21/strided_slice/stack:output:02conv2d_transpose_21/strided_slice/stack_1:output:02conv2d_transpose_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_21/strided_slice|
conv2d_transpose_21/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_21/stack/1|
conv2d_transpose_21/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_21/stack/2|
conv2d_transpose_21/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_21/stack/3?
conv2d_transpose_21/stackPack*conv2d_transpose_21/strided_slice:output:0$conv2d_transpose_21/stack/1:output:0$conv2d_transpose_21/stack/2:output:0$conv2d_transpose_21/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_21/stack?
)conv2d_transpose_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_21/strided_slice_1/stack?
+conv2d_transpose_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_21/strided_slice_1/stack_1?
+conv2d_transpose_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_21/strided_slice_1/stack_2?
#conv2d_transpose_21/strided_slice_1StridedSlice"conv2d_transpose_21/stack:output:02conv2d_transpose_21/strided_slice_1/stack:output:04conv2d_transpose_21/strided_slice_1/stack_1:output:04conv2d_transpose_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_21/strided_slice_1?
3conv2d_transpose_21/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_21/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_21/conv2d_transposeConv2DBackpropInput"conv2d_transpose_21/stack:output:0;conv2d_transpose_21/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_26/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2&
$conv2d_transpose_21/conv2d_transpose?
*conv2d_transpose_21/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_21/BiasAdd/ReadVariableOp?
conv2d_transpose_21/BiasAddBiasAdd-conv2d_transpose_21/conv2d_transpose:output:02conv2d_transpose_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_21/BiasAdd?
conv2d_transpose_21/SeluSelu$conv2d_transpose_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_21/Selu?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_21/Selu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_27/FusedBatchNormV3?
%batch_normalization_27/AssignNewValueAssignVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource4batch_normalization_27/FusedBatchNormV3:batch_mean:07^batch_normalization_27/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_27/AssignNewValue?
'batch_normalization_27/AssignNewValue_1AssignVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_27/FusedBatchNormV3:batch_variance:09^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_27/AssignNewValue_1?
conv2d_transpose_22/ShapeShape+batch_normalization_27/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_22/Shape?
'conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_22/strided_slice/stack?
)conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_22/strided_slice/stack_1?
)conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_22/strided_slice/stack_2?
!conv2d_transpose_22/strided_sliceStridedSlice"conv2d_transpose_22/Shape:output:00conv2d_transpose_22/strided_slice/stack:output:02conv2d_transpose_22/strided_slice/stack_1:output:02conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_22/strided_slice|
conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_22/stack/1|
conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_22/stack/2|
conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_22/stack/3?
conv2d_transpose_22/stackPack*conv2d_transpose_22/strided_slice:output:0$conv2d_transpose_22/stack/1:output:0$conv2d_transpose_22/stack/2:output:0$conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_22/stack?
)conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_22/strided_slice_1/stack?
+conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_22/strided_slice_1/stack_1?
+conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_22/strided_slice_1/stack_2?
#conv2d_transpose_22/strided_slice_1StridedSlice"conv2d_transpose_22/stack:output:02conv2d_transpose_22/strided_slice_1/stack:output:04conv2d_transpose_22/strided_slice_1/stack_1:output:04conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_22/strided_slice_1?
3conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_22/conv2d_transposeConv2DBackpropInput"conv2d_transpose_22/stack:output:0;conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2&
$conv2d_transpose_22/conv2d_transpose?
*conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_22/BiasAdd/ReadVariableOp?
conv2d_transpose_22/BiasAddBiasAdd-conv2d_transpose_22/conv2d_transpose:output:02conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_22/BiasAdd?
conv2d_transpose_22/SeluSelu$conv2d_transpose_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_22/Selu?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_22/Selu:activations:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_28/FusedBatchNormV3?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_28/AssignNewValue?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_28/AssignNewValue_1?
conv2d_transpose_23/ShapeShape+batch_normalization_28/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_23/Shape?
'conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_23/strided_slice/stack?
)conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_23/strided_slice/stack_1?
)conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_23/strided_slice/stack_2?
!conv2d_transpose_23/strided_sliceStridedSlice"conv2d_transpose_23/Shape:output:00conv2d_transpose_23/strided_slice/stack:output:02conv2d_transpose_23/strided_slice/stack_1:output:02conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_23/strided_slice}
conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_23/stack/1}
conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_23/stack/2|
conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_23/stack/3?
conv2d_transpose_23/stackPack*conv2d_transpose_23/strided_slice:output:0$conv2d_transpose_23/stack/1:output:0$conv2d_transpose_23/stack/2:output:0$conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_23/stack?
)conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_23/strided_slice_1/stack?
+conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_23/strided_slice_1/stack_1?
+conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_23/strided_slice_1/stack_2?
#conv2d_transpose_23/strided_slice_1StridedSlice"conv2d_transpose_23/stack:output:02conv2d_transpose_23/strided_slice_1/stack:output:04conv2d_transpose_23/strided_slice_1/stack_1:output:04conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_23/strided_slice_1?
3conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_23/conv2d_transposeConv2DBackpropInput"conv2d_transpose_23/stack:output:0;conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_28/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$conv2d_transpose_23/conv2d_transpose?
*conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_23/BiasAdd/ReadVariableOp?
conv2d_transpose_23/BiasAddBiasAdd-conv2d_transpose_23/conv2d_transpose:output:02conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_23/BiasAdd?
conv2d_transpose_23/TanhTanh$conv2d_transpose_23/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_23/Tanh?

IdentityIdentityconv2d_transpose_23/Tanh:y:0&^batch_normalization_26/AssignNewValue(^batch_normalization_26/AssignNewValue_17^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_1&^batch_normalization_27/AssignNewValue(^batch_normalization_27/AssignNewValue_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_1&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1+^conv2d_transpose_21/BiasAdd/ReadVariableOp4^conv2d_transpose_21/conv2d_transpose/ReadVariableOp+^conv2d_transpose_22/BiasAdd/ReadVariableOp4^conv2d_transpose_22/conv2d_transpose/ReadVariableOp+^conv2d_transpose_23/BiasAdd/ReadVariableOp4^conv2d_transpose_23/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_26/AssignNewValue%batch_normalization_26/AssignNewValue2R
'batch_normalization_26/AssignNewValue_1'batch_normalization_26/AssignNewValue_12p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12N
%batch_normalization_27/AssignNewValue%batch_normalization_27/AssignNewValue2R
'batch_normalization_27/AssignNewValue_1'batch_normalization_27/AssignNewValue_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12X
*conv2d_transpose_21/BiasAdd/ReadVariableOp*conv2d_transpose_21/BiasAdd/ReadVariableOp2j
3conv2d_transpose_21/conv2d_transpose/ReadVariableOp3conv2d_transpose_21/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_22/BiasAdd/ReadVariableOp*conv2d_transpose_22/BiasAdd/ReadVariableOp2j
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp3conv2d_transpose_22/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_23/BiasAdd/ReadVariableOp*conv2d_transpose_23/BiasAdd/ReadVariableOp2j
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp3conv2d_transpose_23/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_27_layer_call_fn_10975763

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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_109743932
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
?7
?

K__inference_sequential_21_layer_call_and_return_conditional_losses_10975207
dense_14_input%
dense_14_10975158:
d??!
dense_14_10975160:
??.
batch_normalization_26_10975164:	?.
batch_normalization_26_10975166:	?.
batch_normalization_26_10975168:	?.
batch_normalization_26_10975170:	?7
conv2d_transpose_21_10975173:@?*
conv2d_transpose_21_10975175:@-
batch_normalization_27_10975178:@-
batch_normalization_27_10975180:@-
batch_normalization_27_10975182:@-
batch_normalization_27_10975184:@6
conv2d_transpose_22_10975187: @*
conv2d_transpose_22_10975189: -
batch_normalization_28_10975192: -
batch_normalization_28_10975194: -
batch_normalization_28_10975196: -
batch_normalization_28_10975198: 6
conv2d_transpose_23_10975201: *
conv2d_transpose_23_10975203:
identity??.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?+conv2d_transpose_21/StatefulPartitionedCall?+conv2d_transpose_22/StatefulPartitionedCall?+conv2d_transpose_23/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCalldense_14_inputdense_14_10975158dense_14_10975160*
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
F__inference_dense_14_layer_call_and_return_conditional_losses_109747302"
 dense_14/StatefulPartitionedCall?
reshape_7/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_reshape_7_layer_call_and_return_conditional_losses_109747502
reshape_7/PartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0batch_normalization_26_10975164batch_normalization_26_10975166batch_normalization_26_10975168batch_normalization_26_10975170*
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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_1097488920
.batch_normalization_26/StatefulPartitionedCall?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_transpose_21_10975173conv2d_transpose_21_10975175*
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
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_109743612-
+conv2d_transpose_21/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0batch_normalization_27_10975178batch_normalization_27_10975180batch_normalization_27_10975182batch_normalization_27_10975184*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1097443720
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_22_10975187conv2d_transpose_22_10975189*
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
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_109745322-
+conv2d_transpose_22/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_28_10975192batch_normalization_28_10975194batch_normalization_28_10975196batch_normalization_28_10975198*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1097460820
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_23_10975201conv2d_transpose_23_10975203*
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
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_109747032-
+conv2d_transpose_23/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_23/StatefulPartitionedCall:output:0/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall,^conv2d_transpose_21/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_14_input
?
?
9__inference_batch_normalization_26_layer_call_fn_10975652

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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_109742662
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
?7
?

K__inference_sequential_21_layer_call_and_return_conditional_losses_10975155
dense_14_input%
dense_14_10975106:
d??!
dense_14_10975108:
??.
batch_normalization_26_10975112:	?.
batch_normalization_26_10975114:	?.
batch_normalization_26_10975116:	?.
batch_normalization_26_10975118:	?7
conv2d_transpose_21_10975121:@?*
conv2d_transpose_21_10975123:@-
batch_normalization_27_10975126:@-
batch_normalization_27_10975128:@-
batch_normalization_27_10975130:@-
batch_normalization_27_10975132:@6
conv2d_transpose_22_10975135: @*
conv2d_transpose_22_10975137: -
batch_normalization_28_10975140: -
batch_normalization_28_10975142: -
batch_normalization_28_10975144: -
batch_normalization_28_10975146: 6
conv2d_transpose_23_10975149: *
conv2d_transpose_23_10975151:
identity??.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?+conv2d_transpose_21/StatefulPartitionedCall?+conv2d_transpose_22/StatefulPartitionedCall?+conv2d_transpose_23/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCalldense_14_inputdense_14_10975106dense_14_10975108*
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
F__inference_dense_14_layer_call_and_return_conditional_losses_109747302"
 dense_14/StatefulPartitionedCall?
reshape_7/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_reshape_7_layer_call_and_return_conditional_losses_109747502
reshape_7/PartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0batch_normalization_26_10975112batch_normalization_26_10975114batch_normalization_26_10975116batch_normalization_26_10975118*
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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_1097476920
.batch_normalization_26/StatefulPartitionedCall?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_transpose_21_10975121conv2d_transpose_21_10975123*
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
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_109743612-
+conv2d_transpose_21/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0batch_normalization_27_10975126batch_normalization_27_10975128batch_normalization_27_10975130batch_normalization_27_10975132*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1097439320
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_22_10975135conv2d_transpose_22_10975137*
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
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_109745322-
+conv2d_transpose_22/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_28_10975140batch_normalization_28_10975142batch_normalization_28_10975144batch_normalization_28_10975146*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1097456420
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_23_10975149conv2d_transpose_23_10975151*
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
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_109747032-
+conv2d_transpose_23/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_23/StatefulPartitionedCall:output:0/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall,^conv2d_transpose_21/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_14_input
?
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975750

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
?
?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10974608

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
?
?
9__inference_batch_normalization_28_layer_call_fn_10975838

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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_109746082
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
?%
?
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_10974532

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
?
?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10974393

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
?
?
0__inference_sequential_21_layer_call_fn_10975299

inputs
unknown:
d??
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

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
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
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_21_layer_call_and_return_conditional_losses_109748132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
6__inference_conv2d_transpose_22_layer_call_fn_10974542

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
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_109745322
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
?
?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10975856

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
?
?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10975874

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
??
?
#__inference__wrapped_model_10974200
dense_14_inputI
5sequential_21_dense_14_matmul_readvariableop_resource:
d??F
6sequential_21_dense_14_biasadd_readvariableop_resource:
??K
<sequential_21_batch_normalization_26_readvariableop_resource:	?M
>sequential_21_batch_normalization_26_readvariableop_1_resource:	?\
Msequential_21_batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_21_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	?e
Jsequential_21_conv2d_transpose_21_conv2d_transpose_readvariableop_resource:@?O
Asequential_21_conv2d_transpose_21_biasadd_readvariableop_resource:@J
<sequential_21_batch_normalization_27_readvariableop_resource:@L
>sequential_21_batch_normalization_27_readvariableop_1_resource:@[
Msequential_21_batch_normalization_27_fusedbatchnormv3_readvariableop_resource:@]
Osequential_21_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:@d
Jsequential_21_conv2d_transpose_22_conv2d_transpose_readvariableop_resource: @O
Asequential_21_conv2d_transpose_22_biasadd_readvariableop_resource: J
<sequential_21_batch_normalization_28_readvariableop_resource: L
>sequential_21_batch_normalization_28_readvariableop_1_resource: [
Msequential_21_batch_normalization_28_fusedbatchnormv3_readvariableop_resource: ]
Osequential_21_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: d
Jsequential_21_conv2d_transpose_23_conv2d_transpose_readvariableop_resource: O
Asequential_21_conv2d_transpose_23_biasadd_readvariableop_resource:
identity??Dsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?Fsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?3sequential_21/batch_normalization_26/ReadVariableOp?5sequential_21/batch_normalization_26/ReadVariableOp_1?Dsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?Fsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?3sequential_21/batch_normalization_27/ReadVariableOp?5sequential_21/batch_normalization_27/ReadVariableOp_1?Dsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Fsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?3sequential_21/batch_normalization_28/ReadVariableOp?5sequential_21/batch_normalization_28/ReadVariableOp_1?8sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOp?Asequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOp?8sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOp?Asequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOp?8sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOp?Asequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOp?-sequential_21/dense_14/BiasAdd/ReadVariableOp?,sequential_21/dense_14/MatMul/ReadVariableOp?
,sequential_21/dense_14/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02.
,sequential_21/dense_14/MatMul/ReadVariableOp?
sequential_21/dense_14/MatMulMatMuldense_14_input4sequential_21/dense_14/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
sequential_21/dense_14/MatMul?
-sequential_21/dense_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_14_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02/
-sequential_21/dense_14/BiasAdd/ReadVariableOp?
sequential_21/dense_14/BiasAddBiasAdd'sequential_21/dense_14/MatMul:product:05sequential_21/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2 
sequential_21/dense_14/BiasAdd?
sequential_21/reshape_7/ShapeShape'sequential_21/dense_14/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_21/reshape_7/Shape?
+sequential_21/reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_21/reshape_7/strided_slice/stack?
-sequential_21/reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_21/reshape_7/strided_slice/stack_1?
-sequential_21/reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_21/reshape_7/strided_slice/stack_2?
%sequential_21/reshape_7/strided_sliceStridedSlice&sequential_21/reshape_7/Shape:output:04sequential_21/reshape_7/strided_slice/stack:output:06sequential_21/reshape_7/strided_slice/stack_1:output:06sequential_21/reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_21/reshape_7/strided_slice?
'sequential_21/reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_21/reshape_7/Reshape/shape/1?
'sequential_21/reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_21/reshape_7/Reshape/shape/2?
'sequential_21/reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2)
'sequential_21/reshape_7/Reshape/shape/3?
%sequential_21/reshape_7/Reshape/shapePack.sequential_21/reshape_7/strided_slice:output:00sequential_21/reshape_7/Reshape/shape/1:output:00sequential_21/reshape_7/Reshape/shape/2:output:00sequential_21/reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_21/reshape_7/Reshape/shape?
sequential_21/reshape_7/ReshapeReshape'sequential_21/dense_14/BiasAdd:output:0.sequential_21/reshape_7/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2!
sequential_21/reshape_7/Reshape?
3sequential_21/batch_normalization_26/ReadVariableOpReadVariableOp<sequential_21_batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype025
3sequential_21/batch_normalization_26/ReadVariableOp?
5sequential_21/batch_normalization_26/ReadVariableOp_1ReadVariableOp>sequential_21_batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5sequential_21/batch_normalization_26/ReadVariableOp_1?
Dsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_21_batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
Fsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_21_batch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
5sequential_21/batch_normalization_26/FusedBatchNormV3FusedBatchNormV3(sequential_21/reshape_7/Reshape:output:0;sequential_21/batch_normalization_26/ReadVariableOp:value:0=sequential_21/batch_normalization_26/ReadVariableOp_1:value:0Lsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 27
5sequential_21/batch_normalization_26/FusedBatchNormV3?
'sequential_21/conv2d_transpose_21/ShapeShape9sequential_21/batch_normalization_26/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2)
'sequential_21/conv2d_transpose_21/Shape?
5sequential_21/conv2d_transpose_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_21/conv2d_transpose_21/strided_slice/stack?
7sequential_21/conv2d_transpose_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_21/conv2d_transpose_21/strided_slice/stack_1?
7sequential_21/conv2d_transpose_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_21/conv2d_transpose_21/strided_slice/stack_2?
/sequential_21/conv2d_transpose_21/strided_sliceStridedSlice0sequential_21/conv2d_transpose_21/Shape:output:0>sequential_21/conv2d_transpose_21/strided_slice/stack:output:0@sequential_21/conv2d_transpose_21/strided_slice/stack_1:output:0@sequential_21/conv2d_transpose_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_21/conv2d_transpose_21/strided_slice?
)sequential_21/conv2d_transpose_21/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_21/conv2d_transpose_21/stack/1?
)sequential_21/conv2d_transpose_21/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_21/conv2d_transpose_21/stack/2?
)sequential_21/conv2d_transpose_21/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_21/conv2d_transpose_21/stack/3?
'sequential_21/conv2d_transpose_21/stackPack8sequential_21/conv2d_transpose_21/strided_slice:output:02sequential_21/conv2d_transpose_21/stack/1:output:02sequential_21/conv2d_transpose_21/stack/2:output:02sequential_21/conv2d_transpose_21/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_21/conv2d_transpose_21/stack?
7sequential_21/conv2d_transpose_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_21/conv2d_transpose_21/strided_slice_1/stack?
9sequential_21/conv2d_transpose_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_21/conv2d_transpose_21/strided_slice_1/stack_1?
9sequential_21/conv2d_transpose_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_21/conv2d_transpose_21/strided_slice_1/stack_2?
1sequential_21/conv2d_transpose_21/strided_slice_1StridedSlice0sequential_21/conv2d_transpose_21/stack:output:0@sequential_21/conv2d_transpose_21/strided_slice_1/stack:output:0Bsequential_21/conv2d_transpose_21/strided_slice_1/stack_1:output:0Bsequential_21/conv2d_transpose_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_21/conv2d_transpose_21/strided_slice_1?
Asequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_21_conv2d_transpose_21_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype02C
Asequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOp?
2sequential_21/conv2d_transpose_21/conv2d_transposeConv2DBackpropInput0sequential_21/conv2d_transpose_21/stack:output:0Isequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOp:value:09sequential_21/batch_normalization_26/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
24
2sequential_21/conv2d_transpose_21/conv2d_transpose?
8sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOpReadVariableOpAsequential_21_conv2d_transpose_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOp?
)sequential_21/conv2d_transpose_21/BiasAddBiasAdd;sequential_21/conv2d_transpose_21/conv2d_transpose:output:0@sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2+
)sequential_21/conv2d_transpose_21/BiasAdd?
&sequential_21/conv2d_transpose_21/SeluSelu2sequential_21/conv2d_transpose_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2(
&sequential_21/conv2d_transpose_21/Selu?
3sequential_21/batch_normalization_27/ReadVariableOpReadVariableOp<sequential_21_batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_21/batch_normalization_27/ReadVariableOp?
5sequential_21/batch_normalization_27/ReadVariableOp_1ReadVariableOp>sequential_21_batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_21/batch_normalization_27/ReadVariableOp_1?
Dsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_21_batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
Fsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_21_batch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
5sequential_21/batch_normalization_27/FusedBatchNormV3FusedBatchNormV34sequential_21/conv2d_transpose_21/Selu:activations:0;sequential_21/batch_normalization_27/ReadVariableOp:value:0=sequential_21/batch_normalization_27/ReadVariableOp_1:value:0Lsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 27
5sequential_21/batch_normalization_27/FusedBatchNormV3?
'sequential_21/conv2d_transpose_22/ShapeShape9sequential_21/batch_normalization_27/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2)
'sequential_21/conv2d_transpose_22/Shape?
5sequential_21/conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_21/conv2d_transpose_22/strided_slice/stack?
7sequential_21/conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_21/conv2d_transpose_22/strided_slice/stack_1?
7sequential_21/conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_21/conv2d_transpose_22/strided_slice/stack_2?
/sequential_21/conv2d_transpose_22/strided_sliceStridedSlice0sequential_21/conv2d_transpose_22/Shape:output:0>sequential_21/conv2d_transpose_22/strided_slice/stack:output:0@sequential_21/conv2d_transpose_22/strided_slice/stack_1:output:0@sequential_21/conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_21/conv2d_transpose_22/strided_slice?
)sequential_21/conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_21/conv2d_transpose_22/stack/1?
)sequential_21/conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2+
)sequential_21/conv2d_transpose_22/stack/2?
)sequential_21/conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential_21/conv2d_transpose_22/stack/3?
'sequential_21/conv2d_transpose_22/stackPack8sequential_21/conv2d_transpose_22/strided_slice:output:02sequential_21/conv2d_transpose_22/stack/1:output:02sequential_21/conv2d_transpose_22/stack/2:output:02sequential_21/conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_21/conv2d_transpose_22/stack?
7sequential_21/conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_21/conv2d_transpose_22/strided_slice_1/stack?
9sequential_21/conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_21/conv2d_transpose_22/strided_slice_1/stack_1?
9sequential_21/conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_21/conv2d_transpose_22/strided_slice_1/stack_2?
1sequential_21/conv2d_transpose_22/strided_slice_1StridedSlice0sequential_21/conv2d_transpose_22/stack:output:0@sequential_21/conv2d_transpose_22/strided_slice_1/stack:output:0Bsequential_21/conv2d_transpose_22/strided_slice_1/stack_1:output:0Bsequential_21/conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_21/conv2d_transpose_22/strided_slice_1?
Asequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_21_conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02C
Asequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOp?
2sequential_21/conv2d_transpose_22/conv2d_transposeConv2DBackpropInput0sequential_21/conv2d_transpose_22/stack:output:0Isequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:09sequential_21/batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
24
2sequential_21/conv2d_transpose_22/conv2d_transpose?
8sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOpAsequential_21_conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOp?
)sequential_21/conv2d_transpose_22/BiasAddBiasAdd;sequential_21/conv2d_transpose_22/conv2d_transpose:output:0@sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2+
)sequential_21/conv2d_transpose_22/BiasAdd?
&sequential_21/conv2d_transpose_22/SeluSelu2sequential_21/conv2d_transpose_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2(
&sequential_21/conv2d_transpose_22/Selu?
3sequential_21/batch_normalization_28/ReadVariableOpReadVariableOp<sequential_21_batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_21/batch_normalization_28/ReadVariableOp?
5sequential_21/batch_normalization_28/ReadVariableOp_1ReadVariableOp>sequential_21_batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_21/batch_normalization_28/ReadVariableOp_1?
Dsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_21_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
Fsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_21_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
5sequential_21/batch_normalization_28/FusedBatchNormV3FusedBatchNormV34sequential_21/conv2d_transpose_22/Selu:activations:0;sequential_21/batch_normalization_28/ReadVariableOp:value:0=sequential_21/batch_normalization_28/ReadVariableOp_1:value:0Lsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 27
5sequential_21/batch_normalization_28/FusedBatchNormV3?
'sequential_21/conv2d_transpose_23/ShapeShape9sequential_21/batch_normalization_28/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2)
'sequential_21/conv2d_transpose_23/Shape?
5sequential_21/conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_21/conv2d_transpose_23/strided_slice/stack?
7sequential_21/conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_21/conv2d_transpose_23/strided_slice/stack_1?
7sequential_21/conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_21/conv2d_transpose_23/strided_slice/stack_2?
/sequential_21/conv2d_transpose_23/strided_sliceStridedSlice0sequential_21/conv2d_transpose_23/Shape:output:0>sequential_21/conv2d_transpose_23/strided_slice/stack:output:0@sequential_21/conv2d_transpose_23/strided_slice/stack_1:output:0@sequential_21/conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_21/conv2d_transpose_23/strided_slice?
)sequential_21/conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_21/conv2d_transpose_23/stack/1?
)sequential_21/conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2+
)sequential_21/conv2d_transpose_23/stack/2?
)sequential_21/conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_21/conv2d_transpose_23/stack/3?
'sequential_21/conv2d_transpose_23/stackPack8sequential_21/conv2d_transpose_23/strided_slice:output:02sequential_21/conv2d_transpose_23/stack/1:output:02sequential_21/conv2d_transpose_23/stack/2:output:02sequential_21/conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'sequential_21/conv2d_transpose_23/stack?
7sequential_21/conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_21/conv2d_transpose_23/strided_slice_1/stack?
9sequential_21/conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_21/conv2d_transpose_23/strided_slice_1/stack_1?
9sequential_21/conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_21/conv2d_transpose_23/strided_slice_1/stack_2?
1sequential_21/conv2d_transpose_23/strided_slice_1StridedSlice0sequential_21/conv2d_transpose_23/stack:output:0@sequential_21/conv2d_transpose_23/strided_slice_1/stack:output:0Bsequential_21/conv2d_transpose_23/strided_slice_1/stack_1:output:0Bsequential_21/conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_21/conv2d_transpose_23/strided_slice_1?
Asequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOpJsequential_21_conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02C
Asequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOp?
2sequential_21/conv2d_transpose_23/conv2d_transposeConv2DBackpropInput0sequential_21/conv2d_transpose_23/stack:output:0Isequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:09sequential_21/batch_normalization_28/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
24
2sequential_21/conv2d_transpose_23/conv2d_transpose?
8sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOpAsequential_21_conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOp?
)sequential_21/conv2d_transpose_23/BiasAddBiasAdd;sequential_21/conv2d_transpose_23/conv2d_transpose:output:0@sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)sequential_21/conv2d_transpose_23/BiasAdd?
&sequential_21/conv2d_transpose_23/TanhTanh2sequential_21/conv2d_transpose_23/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2(
&sequential_21/conv2d_transpose_23/Tanh?

IdentityIdentity*sequential_21/conv2d_transpose_23/Tanh:y:0E^sequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOpG^sequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_14^sequential_21/batch_normalization_26/ReadVariableOp6^sequential_21/batch_normalization_26/ReadVariableOp_1E^sequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOpG^sequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_14^sequential_21/batch_normalization_27/ReadVariableOp6^sequential_21/batch_normalization_27/ReadVariableOp_1E^sequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOpG^sequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_14^sequential_21/batch_normalization_28/ReadVariableOp6^sequential_21/batch_normalization_28/ReadVariableOp_19^sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOpB^sequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOp9^sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOpB^sequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOp9^sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOpB^sequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOp.^sequential_21/dense_14/BiasAdd/ReadVariableOp-^sequential_21/dense_14/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2?
Dsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOpDsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp2?
Fsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1Fsequential_21/batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12j
3sequential_21/batch_normalization_26/ReadVariableOp3sequential_21/batch_normalization_26/ReadVariableOp2n
5sequential_21/batch_normalization_26/ReadVariableOp_15sequential_21/batch_normalization_26/ReadVariableOp_12?
Dsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOpDsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp2?
Fsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1Fsequential_21/batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12j
3sequential_21/batch_normalization_27/ReadVariableOp3sequential_21/batch_normalization_27/ReadVariableOp2n
5sequential_21/batch_normalization_27/ReadVariableOp_15sequential_21/batch_normalization_27/ReadVariableOp_12?
Dsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOpDsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Fsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Fsequential_21/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12j
3sequential_21/batch_normalization_28/ReadVariableOp3sequential_21/batch_normalization_28/ReadVariableOp2n
5sequential_21/batch_normalization_28/ReadVariableOp_15sequential_21/batch_normalization_28/ReadVariableOp_12t
8sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOp8sequential_21/conv2d_transpose_21/BiasAdd/ReadVariableOp2?
Asequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOpAsequential_21/conv2d_transpose_21/conv2d_transpose/ReadVariableOp2t
8sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOp8sequential_21/conv2d_transpose_22/BiasAdd/ReadVariableOp2?
Asequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOpAsequential_21/conv2d_transpose_22/conv2d_transpose/ReadVariableOp2t
8sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOp8sequential_21/conv2d_transpose_23/BiasAdd/ReadVariableOp2?
Asequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOpAsequential_21/conv2d_transpose_23/conv2d_transpose/ReadVariableOp2^
-sequential_21/dense_14/BiasAdd/ReadVariableOp-sequential_21/dense_14/BiasAdd/ReadVariableOp2\
,sequential_21/dense_14/MatMul/ReadVariableOp,sequential_21/dense_14/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_14_input
?
?
&__inference_signature_wrapper_10975254
dense_14_input
unknown:
d??
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

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_109742002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_14_input
?
?
0__inference_sequential_21_layer_call_fn_10974856
dense_14_input
unknown:
d??
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

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_21_layer_call_and_return_conditional_losses_109748132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_14_input
?
?
0__inference_sequential_21_layer_call_fn_10975344

inputs
unknown:
d??
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

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
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
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_21_layer_call_and_return_conditional_losses_109750152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
c
G__inference_reshape_7_layer_call_and_return_conditional_losses_10974750

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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10974889

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
?
?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10975812

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
9__inference_batch_normalization_26_layer_call_fn_10975678

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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_109748892
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
?
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10974266

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
?
?
0__inference_sequential_21_layer_call_fn_10975103
dense_14_input
unknown:
d??
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

unknown_11: @

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_21_layer_call_and_return_conditional_losses_109750152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????d
(
_user_specified_namedense_14_input
?
?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10975794

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
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975696

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
?
?
+__inference_dense_14_layer_call_fn_10975597

inputs
unknown:
d??
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
F__inference_dense_14_layer_call_and_return_conditional_losses_109747302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?%
?
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_10974361

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
ɴ
?
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975466

inputs;
'dense_14_matmul_readvariableop_resource:
d??8
(dense_14_biasadd_readvariableop_resource:
??=
.batch_normalization_26_readvariableop_resource:	??
0batch_normalization_26_readvariableop_1_resource:	?N
?batch_normalization_26_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_21_conv2d_transpose_readvariableop_resource:@?A
3conv2d_transpose_21_biasadd_readvariableop_resource:@<
.batch_normalization_27_readvariableop_resource:@>
0batch_normalization_27_readvariableop_1_resource:@M
?batch_normalization_27_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_22_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_22_biasadd_readvariableop_resource: <
.batch_normalization_28_readvariableop_resource: >
0batch_normalization_28_readvariableop_1_resource: M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_23_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_23_biasadd_readvariableop_resource:
identity??6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_26/ReadVariableOp?'batch_normalization_26/ReadVariableOp_1?6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_27/ReadVariableOp?'batch_normalization_27/ReadVariableOp_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?*conv2d_transpose_21/BiasAdd/ReadVariableOp?3conv2d_transpose_21/conv2d_transpose/ReadVariableOp?*conv2d_transpose_22/BiasAdd/ReadVariableOp?3conv2d_transpose_22/conv2d_transpose/ReadVariableOp?*conv2d_transpose_23/BiasAdd/ReadVariableOp?3conv2d_transpose_23/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/BiasAddk
reshape_7/ShapeShapedense_14/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_7/Shape?
reshape_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_7/strided_slice/stack?
reshape_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_7/strided_slice/stack_1?
reshape_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_7/strided_slice/stack_2?
reshape_7/strided_sliceStridedSlicereshape_7/Shape:output:0&reshape_7/strided_slice/stack:output:0(reshape_7/strided_slice/stack_1:output:0(reshape_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_7/strided_slicex
reshape_7/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_7/Reshape/shape/1x
reshape_7/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_7/Reshape/shape/2y
reshape_7/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_7/Reshape/shape/3?
reshape_7/Reshape/shapePack reshape_7/strided_slice:output:0"reshape_7/Reshape/shape/1:output:0"reshape_7/Reshape/shape/2:output:0"reshape_7/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_7/Reshape/shape?
reshape_7/ReshapeReshapedense_14/BiasAdd:output:0 reshape_7/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_7/Reshape?
%batch_normalization_26/ReadVariableOpReadVariableOp.batch_normalization_26_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_26/ReadVariableOp?
'batch_normalization_26/ReadVariableOp_1ReadVariableOp0batch_normalization_26_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_26/ReadVariableOp_1?
6batch_normalization_26/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_26_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_26_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_26/FusedBatchNormV3FusedBatchNormV3reshape_7/Reshape:output:0-batch_normalization_26/ReadVariableOp:value:0/batch_normalization_26/ReadVariableOp_1:value:0>batch_normalization_26/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_26/FusedBatchNormV3?
conv2d_transpose_21/ShapeShape+batch_normalization_26/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_21/Shape?
'conv2d_transpose_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_21/strided_slice/stack?
)conv2d_transpose_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_21/strided_slice/stack_1?
)conv2d_transpose_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_21/strided_slice/stack_2?
!conv2d_transpose_21/strided_sliceStridedSlice"conv2d_transpose_21/Shape:output:00conv2d_transpose_21/strided_slice/stack:output:02conv2d_transpose_21/strided_slice/stack_1:output:02conv2d_transpose_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_21/strided_slice|
conv2d_transpose_21/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_21/stack/1|
conv2d_transpose_21/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_21/stack/2|
conv2d_transpose_21/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_21/stack/3?
conv2d_transpose_21/stackPack*conv2d_transpose_21/strided_slice:output:0$conv2d_transpose_21/stack/1:output:0$conv2d_transpose_21/stack/2:output:0$conv2d_transpose_21/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_21/stack?
)conv2d_transpose_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_21/strided_slice_1/stack?
+conv2d_transpose_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_21/strided_slice_1/stack_1?
+conv2d_transpose_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_21/strided_slice_1/stack_2?
#conv2d_transpose_21/strided_slice_1StridedSlice"conv2d_transpose_21/stack:output:02conv2d_transpose_21/strided_slice_1/stack:output:04conv2d_transpose_21/strided_slice_1/stack_1:output:04conv2d_transpose_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_21/strided_slice_1?
3conv2d_transpose_21/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype025
3conv2d_transpose_21/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_21/conv2d_transposeConv2DBackpropInput"conv2d_transpose_21/stack:output:0;conv2d_transpose_21/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_26/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
2&
$conv2d_transpose_21/conv2d_transpose?
*conv2d_transpose_21/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*conv2d_transpose_21/BiasAdd/ReadVariableOp?
conv2d_transpose_21/BiasAddBiasAdd-conv2d_transpose_21/conv2d_transpose:output:02conv2d_transpose_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_21/BiasAdd?
conv2d_transpose_21/SeluSelu$conv2d_transpose_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  @2
conv2d_transpose_21/Selu?
%batch_normalization_27/ReadVariableOpReadVariableOp.batch_normalization_27_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_27/ReadVariableOp?
'batch_normalization_27/ReadVariableOp_1ReadVariableOp0batch_normalization_27_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_27/ReadVariableOp_1?
6batch_normalization_27/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_27_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_27_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_27/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_21/Selu:activations:0-batch_normalization_27/ReadVariableOp:value:0/batch_normalization_27/ReadVariableOp_1:value:0>batch_normalization_27/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_27/FusedBatchNormV3?
conv2d_transpose_22/ShapeShape+batch_normalization_27/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_22/Shape?
'conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_22/strided_slice/stack?
)conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_22/strided_slice/stack_1?
)conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_22/strided_slice/stack_2?
!conv2d_transpose_22/strided_sliceStridedSlice"conv2d_transpose_22/Shape:output:00conv2d_transpose_22/strided_slice/stack:output:02conv2d_transpose_22/strided_slice/stack_1:output:02conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_22/strided_slice|
conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_22/stack/1|
conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_22/stack/2|
conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_22/stack/3?
conv2d_transpose_22/stackPack*conv2d_transpose_22/strided_slice:output:0$conv2d_transpose_22/stack/1:output:0$conv2d_transpose_22/stack/2:output:0$conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_22/stack?
)conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_22/strided_slice_1/stack?
+conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_22/strided_slice_1/stack_1?
+conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_22/strided_slice_1/stack_2?
#conv2d_transpose_22/strided_slice_1StridedSlice"conv2d_transpose_22/stack:output:02conv2d_transpose_22/strided_slice_1/stack:output:04conv2d_transpose_22/strided_slice_1/stack_1:output:04conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_22/strided_slice_1?
3conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype025
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_22/conv2d_transposeConv2DBackpropInput"conv2d_transpose_22/stack:output:0;conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_27/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2&
$conv2d_transpose_22/conv2d_transpose?
*conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_22/BiasAdd/ReadVariableOp?
conv2d_transpose_22/BiasAddBiasAdd-conv2d_transpose_22/conv2d_transpose:output:02conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_22/BiasAdd?
conv2d_transpose_22/SeluSelu$conv2d_transpose_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_transpose_22/Selu?
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_28/ReadVariableOp?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_28/ReadVariableOp_1?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_22/Selu:activations:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@ : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_28/FusedBatchNormV3?
conv2d_transpose_23/ShapeShape+batch_normalization_28/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_23/Shape?
'conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_23/strided_slice/stack?
)conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_23/strided_slice/stack_1?
)conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_23/strided_slice/stack_2?
!conv2d_transpose_23/strided_sliceStridedSlice"conv2d_transpose_23/Shape:output:00conv2d_transpose_23/strided_slice/stack:output:02conv2d_transpose_23/strided_slice/stack_1:output:02conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_23/strided_slice}
conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_23/stack/1}
conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_23/stack/2|
conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_23/stack/3?
conv2d_transpose_23/stackPack*conv2d_transpose_23/strided_slice:output:0$conv2d_transpose_23/stack/1:output:0$conv2d_transpose_23/stack/2:output:0$conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_23/stack?
)conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_23/strided_slice_1/stack?
+conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_23/strided_slice_1/stack_1?
+conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_23/strided_slice_1/stack_2?
#conv2d_transpose_23/strided_slice_1StridedSlice"conv2d_transpose_23/stack:output:02conv2d_transpose_23/strided_slice_1/stack:output:04conv2d_transpose_23/strided_slice_1/stack_1:output:04conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_23/strided_slice_1?
3conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_23/conv2d_transposeConv2DBackpropInput"conv2d_transpose_23/stack:output:0;conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_28/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2&
$conv2d_transpose_23/conv2d_transpose?
*conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_23/BiasAdd/ReadVariableOp?
conv2d_transpose_23/BiasAddBiasAdd-conv2d_transpose_23/conv2d_transpose:output:02conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_23/BiasAdd?
conv2d_transpose_23/TanhTanh$conv2d_transpose_23/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_transpose_23/Tanh?
IdentityIdentityconv2d_transpose_23/Tanh:y:07^batch_normalization_26/FusedBatchNormV3/ReadVariableOp9^batch_normalization_26/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_26/ReadVariableOp(^batch_normalization_26/ReadVariableOp_17^batch_normalization_27/FusedBatchNormV3/ReadVariableOp9^batch_normalization_27/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_27/ReadVariableOp(^batch_normalization_27/ReadVariableOp_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1+^conv2d_transpose_21/BiasAdd/ReadVariableOp4^conv2d_transpose_21/conv2d_transpose/ReadVariableOp+^conv2d_transpose_22/BiasAdd/ReadVariableOp4^conv2d_transpose_22/conv2d_transpose/ReadVariableOp+^conv2d_transpose_23/BiasAdd/ReadVariableOp4^conv2d_transpose_23/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_26/FusedBatchNormV3/ReadVariableOp6batch_normalization_26/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_26/FusedBatchNormV3/ReadVariableOp_18batch_normalization_26/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_26/ReadVariableOp%batch_normalization_26/ReadVariableOp2R
'batch_normalization_26/ReadVariableOp_1'batch_normalization_26/ReadVariableOp_12p
6batch_normalization_27/FusedBatchNormV3/ReadVariableOp6batch_normalization_27/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_27/FusedBatchNormV3/ReadVariableOp_18batch_normalization_27/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_27/ReadVariableOp%batch_normalization_27/ReadVariableOp2R
'batch_normalization_27/ReadVariableOp_1'batch_normalization_27/ReadVariableOp_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12X
*conv2d_transpose_21/BiasAdd/ReadVariableOp*conv2d_transpose_21/BiasAdd/ReadVariableOp2j
3conv2d_transpose_21/conv2d_transpose/ReadVariableOp3conv2d_transpose_21/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_22/BiasAdd/ReadVariableOp*conv2d_transpose_22/BiasAdd/ReadVariableOp2j
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp3conv2d_transpose_22/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_23/BiasAdd/ReadVariableOp*conv2d_transpose_23/BiasAdd/ReadVariableOp2j
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp3conv2d_transpose_23/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10974437

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
?
F__inference_dense_14_layer_call_and_return_conditional_losses_10974730

inputs2
matmul_readvariableop_resource:
d??/
biasadd_readvariableop_resource:
??
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d??*
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
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10974769

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
?
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975714

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
?
?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10974564

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
?
c
G__inference_reshape_7_layer_call_and_return_conditional_losses_10975626

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
?6
?

K__inference_sequential_21_layer_call_and_return_conditional_losses_10975015

inputs%
dense_14_10974966:
d??!
dense_14_10974968:
??.
batch_normalization_26_10974972:	?.
batch_normalization_26_10974974:	?.
batch_normalization_26_10974976:	?.
batch_normalization_26_10974978:	?7
conv2d_transpose_21_10974981:@?*
conv2d_transpose_21_10974983:@-
batch_normalization_27_10974986:@-
batch_normalization_27_10974988:@-
batch_normalization_27_10974990:@-
batch_normalization_27_10974992:@6
conv2d_transpose_22_10974995: @*
conv2d_transpose_22_10974997: -
batch_normalization_28_10975000: -
batch_normalization_28_10975002: -
batch_normalization_28_10975004: -
batch_normalization_28_10975006: 6
conv2d_transpose_23_10975009: *
conv2d_transpose_23_10975011:
identity??.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?+conv2d_transpose_21/StatefulPartitionedCall?+conv2d_transpose_22/StatefulPartitionedCall?+conv2d_transpose_23/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_10974966dense_14_10974968*
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
F__inference_dense_14_layer_call_and_return_conditional_losses_109747302"
 dense_14/StatefulPartitionedCall?
reshape_7/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_reshape_7_layer_call_and_return_conditional_losses_109747502
reshape_7/PartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0batch_normalization_26_10974972batch_normalization_26_10974974batch_normalization_26_10974976batch_normalization_26_10974978*
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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_1097488920
.batch_normalization_26/StatefulPartitionedCall?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_transpose_21_10974981conv2d_transpose_21_10974983*
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
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_109743612-
+conv2d_transpose_21/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0batch_normalization_27_10974986batch_normalization_27_10974988batch_normalization_27_10974990batch_normalization_27_10974992*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1097443720
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_22_10974995conv2d_transpose_22_10974997*
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
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_109745322-
+conv2d_transpose_22/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_28_10975000batch_normalization_28_10975002batch_normalization_28_10975004batch_normalization_28_10975006*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1097460820
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_23_10975009conv2d_transpose_23_10975011*
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
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_109747032-
+conv2d_transpose_23/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_23/StatefulPartitionedCall:output:0/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall,^conv2d_transpose_21/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
H
,__inference_reshape_7_layer_call_fn_10975612

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
GPU2*0J 8? *P
fKRI
G__inference_reshape_7_layer_call_and_return_conditional_losses_109747502
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
?6
?

K__inference_sequential_21_layer_call_and_return_conditional_losses_10974813

inputs%
dense_14_10974731:
d??!
dense_14_10974733:
??.
batch_normalization_26_10974770:	?.
batch_normalization_26_10974772:	?.
batch_normalization_26_10974774:	?.
batch_normalization_26_10974776:	?7
conv2d_transpose_21_10974779:@?*
conv2d_transpose_21_10974781:@-
batch_normalization_27_10974784:@-
batch_normalization_27_10974786:@-
batch_normalization_27_10974788:@-
batch_normalization_27_10974790:@6
conv2d_transpose_22_10974793: @*
conv2d_transpose_22_10974795: -
batch_normalization_28_10974798: -
batch_normalization_28_10974800: -
batch_normalization_28_10974802: -
batch_normalization_28_10974804: 6
conv2d_transpose_23_10974807: *
conv2d_transpose_23_10974809:
identity??.batch_normalization_26/StatefulPartitionedCall?.batch_normalization_27/StatefulPartitionedCall?.batch_normalization_28/StatefulPartitionedCall?+conv2d_transpose_21/StatefulPartitionedCall?+conv2d_transpose_22/StatefulPartitionedCall?+conv2d_transpose_23/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_10974731dense_14_10974733*
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
F__inference_dense_14_layer_call_and_return_conditional_losses_109747302"
 dense_14/StatefulPartitionedCall?
reshape_7/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *P
fKRI
G__inference_reshape_7_layer_call_and_return_conditional_losses_109747502
reshape_7/PartitionedCall?
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall"reshape_7/PartitionedCall:output:0batch_normalization_26_10974770batch_normalization_26_10974772batch_normalization_26_10974774batch_normalization_26_10974776*
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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_1097476920
.batch_normalization_26/StatefulPartitionedCall?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_26/StatefulPartitionedCall:output:0conv2d_transpose_21_10974779conv2d_transpose_21_10974781*
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
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_109743612-
+conv2d_transpose_21/StatefulPartitionedCall?
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0batch_normalization_27_10974784batch_normalization_27_10974786batch_normalization_27_10974788batch_normalization_27_10974790*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_1097439320
.batch_normalization_27/StatefulPartitionedCall?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_27/StatefulPartitionedCall:output:0conv2d_transpose_22_10974793conv2d_transpose_22_10974795*
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
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_109745322-
+conv2d_transpose_22/StatefulPartitionedCall?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_28_10974798batch_normalization_28_10974800batch_normalization_28_10974802batch_normalization_28_10974804*
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
GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1097456420
.batch_normalization_28/StatefulPartitionedCall?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv2d_transpose_23_10974807conv2d_transpose_23_10974809*
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
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_109747032-
+conv2d_transpose_23/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_23/StatefulPartitionedCall:output:0/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall,^conv2d_transpose_21/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????d: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?Z
?
$__inference__traced_restore_10976027
file_prefix4
 assignvariableop_dense_14_kernel:
d??0
 assignvariableop_1_dense_14_bias:
??>
/assignvariableop_2_batch_normalization_26_gamma:	?=
.assignvariableop_3_batch_normalization_26_beta:	?D
5assignvariableop_4_batch_normalization_26_moving_mean:	?H
9assignvariableop_5_batch_normalization_26_moving_variance:	?H
-assignvariableop_6_conv2d_transpose_21_kernel:@?9
+assignvariableop_7_conv2d_transpose_21_bias:@=
/assignvariableop_8_batch_normalization_27_gamma:@<
.assignvariableop_9_batch_normalization_27_beta:@D
6assignvariableop_10_batch_normalization_27_moving_mean:@H
:assignvariableop_11_batch_normalization_27_moving_variance:@H
.assignvariableop_12_conv2d_transpose_22_kernel: @:
,assignvariableop_13_conv2d_transpose_22_bias: >
0assignvariableop_14_batch_normalization_28_gamma: =
/assignvariableop_15_batch_normalization_28_beta: D
6assignvariableop_16_batch_normalization_28_moving_mean: H
:assignvariableop_17_batch_normalization_28_moving_variance: H
.assignvariableop_18_conv2d_transpose_23_kernel: :
,assignvariableop_19_conv2d_transpose_23_bias:
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_26_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_26_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_26_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_26_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_21_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_21_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_27_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_27_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_27_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_27_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_conv2d_transpose_22_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_conv2d_transpose_22_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_28_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_28_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_28_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_28_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_conv2d_transpose_23_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_conv2d_transpose_23_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20?
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22(
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
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
dense_14_input7
 serving_default_dense_14_input:0?????????dQ
conv2d_transpose_23:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?Y
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
		variables

trainable_variables
regularization_losses
	keras_api

signatures
r__call__
*s&call_and_return_all_conditional_losses
t_default_save_signature"?U
_tf_keras_sequential?U{"name": "sequential_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_14_input"}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_7", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16, 128]}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_21", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_23", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100]}, "float32", "dense_14_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_14_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "reshape_7", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16, 128]}}, "shared_object_id": 4}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_21", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 12}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 20}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 25}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_23", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 28}]}}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 32768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "reshape_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_7", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 16, 128]}}, "shared_object_id": 4}
?

axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
 	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
?

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
{__call__
*|&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_transpose_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_21", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
?

'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-trainable_variables
.regularization_losses
/	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_transpose_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}
?

6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
?

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_transpose_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_23", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
?
0
1
2
3
4
5
!6
"7
(8
)9
*10
+11
012
113
714
815
916
:17
?18
@19"
trackable_list_wrapper
?
0
1
2
3
!4
"5
(6
)7
08
19
710
811
?12
@13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Emetrics
		variables

Flayers
Gnon_trainable_variables
Hlayer_metrics

trainable_variables
Ilayer_regularization_losses
regularization_losses
r__call__
t_default_save_signature
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!
d??2dense_14/kernel
:??2dense_14/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jmetrics

Klayers
Lnon_trainable_variables
	variables
Mlayer_metrics
trainable_variables
Nlayer_regularization_losses
regularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ometrics

Players
Qnon_trainable_variables
	variables
Rlayer_metrics
trainable_variables
Slayer_regularization_losses
regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_26/gamma
*:(?2batch_normalization_26/beta
3:1? (2"batch_normalization_26/moving_mean
7:5? (2&batch_normalization_26/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tmetrics

Ulayers
Vnon_trainable_variables
	variables
Wlayer_metrics
trainable_variables
Xlayer_regularization_losses
regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
5:3@?2conv2d_transpose_21/kernel
&:$@2conv2d_transpose_21/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ymetrics

Zlayers
[non_trainable_variables
#	variables
\layer_metrics
$trainable_variables
]layer_regularization_losses
%regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_27/gamma
):'@2batch_normalization_27/beta
2:0@ (2"batch_normalization_27/moving_mean
6:4@ (2&batch_normalization_27/moving_variance
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^metrics

_layers
`non_trainable_variables
,	variables
alayer_metrics
-trainable_variables
blayer_regularization_losses
.regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
4:2 @2conv2d_transpose_22/kernel
&:$ 2conv2d_transpose_22/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
cmetrics

dlayers
enon_trainable_variables
2	variables
flayer_metrics
3trainable_variables
glayer_regularization_losses
4regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_28/gamma
):' 2batch_normalization_28/beta
2:0  (2"batch_normalization_28/moving_mean
6:4  (2&batch_normalization_28/moving_variance
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hmetrics

ilayers
jnon_trainable_variables
;	variables
klayer_metrics
<trainable_variables
llayer_regularization_losses
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2 2conv2d_transpose_23/kernel
&:$2conv2d_transpose_23/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mmetrics

nlayers
onon_trainable_variables
A	variables
player_metrics
Btrainable_variables
qlayer_regularization_losses
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
J
0
1
*2
+3
94
:5"
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
0
1"
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
*0
+1"
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
90
:1"
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
?2?
0__inference_sequential_21_layer_call_fn_10974856
0__inference_sequential_21_layer_call_fn_10975299
0__inference_sequential_21_layer_call_fn_10975344
0__inference_sequential_21_layer_call_fn_10975103?
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
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975466
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975588
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975155
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975207?
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
#__inference__wrapped_model_10974200?
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
annotations? *-?*
(?%
dense_14_input?????????d
?2?
+__inference_dense_14_layer_call_fn_10975597?
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
F__inference_dense_14_layer_call_and_return_conditional_losses_10975607?
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
,__inference_reshape_7_layer_call_fn_10975612?
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
G__inference_reshape_7_layer_call_and_return_conditional_losses_10975626?
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
9__inference_batch_normalization_26_layer_call_fn_10975639
9__inference_batch_normalization_26_layer_call_fn_10975652
9__inference_batch_normalization_26_layer_call_fn_10975665
9__inference_batch_normalization_26_layer_call_fn_10975678?
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
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975696
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975714
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975732
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975750?
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
6__inference_conv2d_transpose_21_layer_call_fn_10974371?
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
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_10974361?
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
9__inference_batch_normalization_27_layer_call_fn_10975763
9__inference_batch_normalization_27_layer_call_fn_10975776?
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
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10975794
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10975812?
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
6__inference_conv2d_transpose_22_layer_call_fn_10974542?
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
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_10974532?
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
9__inference_batch_normalization_28_layer_call_fn_10975825
9__inference_batch_normalization_28_layer_call_fn_10975838?
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
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10975856
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10975874?
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
6__inference_conv2d_transpose_23_layer_call_fn_10974713?
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
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_10974703?
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
&__inference_signature_wrapper_10975254dense_14_input"?
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
#__inference__wrapped_model_10974200?!"()*+01789:?@7?4
-?*
(?%
dense_14_input?????????d
? "S?P
N
conv2d_transpose_237?4
conv2d_transpose_23????????????
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975696?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975714?N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975732t<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_10975750t<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
9__inference_batch_normalization_26_layer_call_fn_10975639?N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
9__inference_batch_normalization_26_layer_call_fn_10975652?N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
9__inference_batch_normalization_26_layer_call_fn_10975665g<?9
2?/
)?&
inputs??????????
p 
? "!????????????
9__inference_batch_normalization_26_layer_call_fn_10975678g<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10975794?()*+M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_10975812?()*+M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
9__inference_batch_normalization_27_layer_call_fn_10975763?()*+M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_27_layer_call_fn_10975776?()*+M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10975856?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_10975874?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
9__inference_batch_normalization_28_layer_call_fn_10975825?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
9__inference_batch_normalization_28_layer_call_fn_10975838?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
Q__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_10974361?!"J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
6__inference_conv2d_transpose_21_layer_call_fn_10974371?!"J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
Q__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_10974532?01I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_conv2d_transpose_22_layer_call_fn_10974542?01I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
Q__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_10974703??@I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
6__inference_conv2d_transpose_23_layer_call_fn_10974713??@I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
F__inference_dense_14_layer_call_and_return_conditional_losses_10975607^/?,
%?"
 ?
inputs?????????d
? "'?$
?
0???????????
? ?
+__inference_dense_14_layer_call_fn_10975597Q/?,
%?"
 ?
inputs?????????d
? "?????????????
G__inference_reshape_7_layer_call_and_return_conditional_losses_10975626c1?.
'?$
"?
inputs???????????
? ".?+
$?!
0??????????
? ?
,__inference_reshape_7_layer_call_fn_10975612V1?.
'?$
"?
inputs???????????
? "!????????????
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975155?!"()*+01789:?@??<
5?2
(?%
dense_14_input?????????d
p 

 
? "??<
5?2
0+???????????????????????????
? ?
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975207?!"()*+01789:?@??<
5?2
(?%
dense_14_input?????????d
p

 
? "??<
5?2
0+???????????????????????????
? ?
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975466?!"()*+01789:?@7?4
-?*
 ?
inputs?????????d
p 

 
? "/?,
%?"
0???????????
? ?
K__inference_sequential_21_layer_call_and_return_conditional_losses_10975588?!"()*+01789:?@7?4
-?*
 ?
inputs?????????d
p

 
? "/?,
%?"
0???????????
? ?
0__inference_sequential_21_layer_call_fn_10974856?!"()*+01789:?@??<
5?2
(?%
dense_14_input?????????d
p 

 
? "2?/+????????????????????????????
0__inference_sequential_21_layer_call_fn_10975103?!"()*+01789:?@??<
5?2
(?%
dense_14_input?????????d
p

 
? "2?/+????????????????????????????
0__inference_sequential_21_layer_call_fn_10975299?!"()*+01789:?@7?4
-?*
 ?
inputs?????????d
p 

 
? "2?/+????????????????????????????
0__inference_sequential_21_layer_call_fn_10975344?!"()*+01789:?@7?4
-?*
 ?
inputs?????????d
p

 
? "2?/+????????????????????????????
&__inference_signature_wrapper_10975254?!"()*+01789:?@I?F
? 
??<
:
dense_14_input(?%
dense_14_input?????????d"S?P
N
conv2d_transpose_237?4
conv2d_transpose_23???????????