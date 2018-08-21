
if isunix || ismac
    cmd = 'cp -a utils/training_code/vllab_dag_loss.m matconvnet/matlab/+dagnn';
else % ispc
    cmd = 'copy utils/training_code/vllab_dag_loss.m matconvnet/matlab/+dagnn';
end
fprintf('%s\n', cmd);
system(cmd);


addpath('matconvnet/matlab');
vl_compilenn('enableGPu', true, ...
    'enableCudnn', true, ...
    'cudnnRoot', '~/cudnn5.1');

