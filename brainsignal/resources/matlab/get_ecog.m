function get_ecog_data(obj_path, out_path)
    libdir = mfilename('fullpath');
    [libdir] = fileparts(libdir);

    repos = dir(libdir);
    repos = repos(~ismember({repos.name},{'.','..'}));

    for i=1:length(repos)
        repo = repos(i);
        if repo.isdir
            addpath(fullfile(repo.folder, repo.name));
        end
    end

    load(obj_path);
    
    s = struct;
    names = fieldnames(obj);
    for k=1:length(names)
        s.(names{k}) = obj.(names{k});
    end

    s = structify(s);

    save(out_path, 's', '-v7.3');

end


function s = structify(s)
    if isstruct(s) && length(s) == 1
        names = fieldnames(s);
        for k=1:length(names)
            s.(names{k}) = structify(s.(names{k}));
        end
    elseif iscell(s)
        for k=1:length(s)
            s{k} = structify(s{k});
        end
    elseif istable(s)
        s = table2struct(s);
    elseif isa(s, 'containers.Map')
        s_ = struct;
        s_.keys = s.keys;
        s_.values = s.values;
        s = s_;
    end
end


