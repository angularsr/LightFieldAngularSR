function imdb = loadIM()

    %% initialize opts
    opts = init_opts();

    addpath(genpath('utils'));

    %% load imdb
    imdb_filename = fullfile('imdb', sprintf('imdb_%s.mat', opts.data_name));
    if( ~exist(imdb_filename, 'file') )
        make_imdb(imdb_filename, opts);
    end
    fprintf('Load data %s\n', imdb_filename);
    imdb = load(imdb_filename);
    img_no = size(imdb(1).images.set, 2);
    imdb.images.set = imdb.images.set(1,[1:img_no]);
    imdb.images.filename = imdb.images.filename([1:img_no],1);
    imdb.images.img = batch_imread(imdb.images.filename, img_no);

    fprintf('Pre-load all images...\n');

