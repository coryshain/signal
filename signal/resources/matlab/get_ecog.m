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
    s.raw = obj.for_preproc.elec_data_raw;
    s.raw_file_paths = obj.raw_file_name;
    s.freq = obj.for_preproc.sample_freq_raw;
    s.time = (1: size(s.raw, 2)) / s.freq;
    s.subject = obj.subject;
    s.experiment = obj.experiment;
    s.channel_names = obj.elec_ch_label;
    s.channel_types = obj.elec_ch_type;
    s.channel_indices = obj.elec_ch;
    s.bad_channels = struct;
    s.bad_channels.auto = obj.elec_ch_prelim_deselect;
    s.bad_channels.manual = obj.elec_ch_user_deselect;
    s.anatomy = obj.anatomy;
    s.events_table = obj.events_table;
    if isprop(obj, 'langloc_save_path')
        s.langloc_save_path = obj.langloc_save_path;
    else
	    s.langloc_save_path = NaN;
    end
    if isprop(obj, 's_vs_n_ops')
        s.s_vs_n_ops = obj.s_vs_n_ops;
    else
	    s.s_vs_n_ops = NaN;
    end
    if isprop(obj, 's_vs_n_sig')
        s.s_vs_n_sig = obj.s_vs_n_sig;
    else
	    s.s_vs_n_sig = NaN;
    end
    if isprop(obj, 's_vs_n_p_ratio')
        s.s_vs_n_p_ratio = obj.s_vs_n_p_ratio;
    else
	    s.s_vs_n_p_ratio = NaN;
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


