function preprocess_trajectories()

    %% read data faces
    fns = dir('~/projects/rogerface/data/*.csv');
    TT = [];ss=0;
    fprintf('loading data faces\n')
    for f=1:numel(fns)
        T = readtable(fullfile(fns(f).folder,fns(f).name),'FileType','text');
        Td = T(strcmp(T.trial_type,'survey-html-form'),:);
        if ~isempty(Td)
            ppinfo = struct2table(jsondecode(strrep(Td.responses{1},'pointer','cursor')),'asarray',1);
            ppinfo.swaporder(:) = contains(Td.responses{1},'cursor');
            T = T(strcmp(T.trial_type,'mousetracking'),:);
            if size(T,1)==512 %only take full datasets
                ss=ss+1;
                T2 = table();
                T2.stimulus = T.stimulus;
                T2.x_position = T.x_position;
                T2.y_position = T.y_position;
                T2.mice_times = T.mice_times;
                T2.nRecordings = T.nRecordings;
                T2.trialnr = (1:512)';
                T2.blocknr = ceil(T2.trialnr./128);
                T2.subject(:) = ss;
                T2 = [T2 repmat(ppinfo,size(T2,1),1)];
                TT = [TT; T2];
            end
        end
    end
    fprintf('loading done\n')

    % add some info
    [~,~,TT.stimulusnr] = unique(TT.stimulus); %add stimulus number (alphabetical)
    TT.isextraface = contains(TT.stimulus,'extra_face');
    TT.isface = contains(TT.stimulus,'face');
    TT.isobject = contains(TT.stimulus,'match');
    TT.islookalike = ~(TT.isface | TT.isobject);
    TT.correct_resp = 1+TT.isface;
    newnumbers = [reshape([33:64; 65:96],[],1)' 97:128 1:32]';
    TT.sortedstimulusnr = newnumbers(TT.stimulusnr);
    %groupsummary(TT,{'sortedstimulusnr','stimulus','stimulusnr'});

    TT = fun2_preproc_traj(TT);

    % save processed data
    fprintf('saving\n')
    save('trajectories-faces.mat','TT','-v7.3')
    fprintf('done\n')


    %% read data
    fns = dir('~/projects/roger3/data/*.csv');
    TT = [];ss=0;
    fprintf('loading data animals\n')
    for f=1:numel(fns)
        T = readtable(fullfile(fns(f).folder,fns(f).name),'FileType','text');
        Td = T(strcmp(T.trial_type,'survey-html-form'),:);
        if ~isempty(Td)
            ppinfo = struct2table(jsondecode(strrep(strrep(Td.responses{1},'pointer','cursor'),'cursor1','cursor')),'asarray',1);
            ppinfo.swaporder(:) = contains(Td.responses{1},'cursor1');
            T = T(strcmp(T.trial_type,'mousetracking'),:);
            if size(T,1)==144 %only take full datasets
                ss=ss+1;
                T2 = table();
                T2.stimulus = T.stimulus;
                T2.x_position = T.x_position;
                T2.y_position = T.y_position;
                T2.mice_times = T.mice_times;
                T2.nRecordings = T.nRecordings;
                T2.trialnr = (1:144)';
                T2.blocknr = ceil(T2.trialnr./36);
                T2.subject(:) = ss;
                T2 = [T2 repmat(ppinfo,size(T2,1),1)];
                TT = [TT; T2];
            end
        end
    end
    fprintf('loading done\n')

    % add some info
    [~,~,TT.stimulusnr] = unique(TT.stimulus); %add stimulus number (alphabetical)
    TT.isanimal = contains(TT.stimulus,'_a');
    TT.islookalike = contains(TT.stimulus,'_l');
    TT.isobject = contains(TT.stimulus,'_o');
    TT.correct_resp = 2-TT.isanimal;
    newnumbers = [10:18 1:9 19:36]';
    TT.sortedstimulusnr = newnumbers(TT.stimulusnr);
    %groupsummary(TT,{'sortedstimulusnr','stimulus','stimulusnr'});

    TT = fun2_preproc_traj(TT);

    % save processed data
    fprintf('saving\n')
    save('trajectories-animals.mat','TT','-v7.3')
    fprintf('done\n')

end

function TT = fun2_preproc_traj(TT)

    %read position data into vectors
    fprintf('reading position data')
    TT.xpos = cellfun(@(x) cellfun(@str2double,strsplit(x,',','CollapseDelimiters',0)), TT.x_position,'UniformOutput',0);fprintf('.');
    TT.ypos = cellfun(@(x) cellfun(@str2double,strsplit(x,',','CollapseDelimiters',0)), TT.y_position,'UniformOutput',0);fprintf('.');
    TT.time = cellfun(@(x) cellfun(@str2double,strsplit(x,',','CollapseDelimiters',0)), TT.mice_times,'UniformOutput',0);fprintf('.');
    TT.x_position = [];TT.y_position = [];TT.mice_times = [];
    fprintf('done\n')

    %realign to screen (0-centre)
    TT.xpos = cellfun(@(x) x-500, TT.xpos,'UniformOutput',0);
    TT.ypos = cellfun(@(x) 600-x, TT.ypos,'UniformOutput',0);

    %flip xpos for 2nd half
    TT.xpos(TT.blocknr>2 & ~TT.swaporder) = cellfun(@(x) -x, TT.xpos(TT.blocknr>2 & ~TT.swaporder),'UniformOutput',0);
    TT.xpos(TT.blocknr<=2 & TT.swaporder) = cellfun(@(x) -x, TT.xpos(TT.blocknr<=2 & TT.swaporder),'UniformOutput',0);

    %get final pos
    TT.finalxpos = cellfun(@(x) x(end), TT.xpos);
    TT.finalypos = cellfun(@(x) x(end), TT.ypos);

    %remove trials where pp did not move more than 50px and have at least 10 measurements
    TT.isvalidtrial = (abs(TT.finalxpos-500)>50 | abs(600-TT.finalypos)>50) & cellfun(@(x) sum(~isnan(x)),TT.xpos)>10;

    TT.correct = (TT.finalxpos>0) == (TT.correct_resp-1);

    %keep only subjects that have more than 100 valid trials
    x = groupsummary(TT,'subject','mean',{'isvalidtrial','correct'});
    x.include = double(x.mean_isvalidtrial>.7 & x.mean_correct>.7);
    TT.isvalidsubject = x.include(TT.subject);
    TT.include = TT.isvalidsubject & TT.isvalidtrial;
    TT.exclude = ~TT.include;

    % interpolate timeseries
    fprintf('interpolating\n')
    newtv = 0:800;
    XPOS = TT.xpos;
    YPOS = TT.ypos;
    TIME = TT.time;
    valid = TT.isvalidtrial;
    ntrial = numel(TIME);
    [xposi,yposi,timei] = deal(zeros(ntrial,numel(newtv)));
    for i=1:size(TIME,1)
        if valid(i)
            [tv,idx] = unique(TIME{i});
            xi = interp1(tv,XPOS{i}(idx),newtv,'linear','extrap');
            xi(isnan(xi)) = xi(find(~isnan(xi),1));
            yi = interp1(tv,YPOS{i}(idx),newtv,'linear','extrap');
            yi(isnan(yi)) = yi(find(~isnan(yi),1));
            xposi(i,:) = xi;
            yposi(i,:) = yi;
            timei(i,:) = newtv;
        end
    end
    TT.xposi = xposi;
    TT.yposi = yposi;
    TT.timei = timei;
    TT.xpos=[];TT.ypos=[];TT.time=[];
    fprintf('done\n')

end