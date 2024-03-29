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
    names = fieldnames(s);
    for k=1:length(names)
        val = s.(names{k});
        if istable(val)
            s.(names{k}) = table2struct(val);
        elseif isa(val, 'containers.Map')
            val_ = struct;
            val_.keys = val.keys;
            val_.values = val.values;
            s.(names{k}) = val_;
        elseif isstruct(val)
            val = structify(val);
            s.(names{k}) = val;
        end
    end
end


