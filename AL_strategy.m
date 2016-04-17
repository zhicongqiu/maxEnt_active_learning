function  sample2label = AL_strategy(PMFunknown,PMFrare,PMFnormal,index_set,active_set_rare,active_set_normal,what)
%input:
%what: 
%0 for top-level uncertainty sampling 
%1 for for suspicious sampling, 
%2 for all-level uncertainty sampling
%3 for most likely unknown sampling (=0 if no labeled rare sample)
%-1 for random sampling
%output:
%sample2label: sample chosen for labeling
%%%%%%%%%%%%%%%%%%%%

if what==3 && sum(active_set_rare)==0
    what = 1;
end

if what==0 %uncertainty sampling (pick highest entropy sample)
    ent_temp = 0;
    for i=1:length(index_set)
        if PMFunknown(index_set(i))~=0 ||PMFunknown(index_set(i))~=1
            temp = PMFunknown(index_set(i))*log(PMFunknown(index_set(i)))...
                + (1-PMFunknown(index_set(i)))*log((1-PMFunknown(index_set(i))));
            temp = -1*temp;
        else
            temp = 0;
        end
        if temp>ent_temp
            ent_temp = temp;
            sample2label = index_set(i);
        end
    end
    
elseif what==1 %suspicious sampling
    sus_temp = 0;
    for i=1:length(index_set)
        if PMFunknown(index_set(i))>sus_temp
            sus_temp = PMFunknown(index_set(i));
            sample2label = index_set(i);
        end
    end
elseif what==2
    max_ent = 0;
    tempR = find(active_set_rare==1);
    tempN = find(active_set_normal==1);
    for i=1:length(index_set)
        ent_temp = 0;
        if PMFunknown(index_set(i))~=0 && PMFunknown(index_set(i))~=1
            if length(tempR)>=1
                for j=1:length(tempR)
                    if PMFrare(index_set(i),tempR(j))~=0
                        ent_temp = ent_temp-...
                            PMFunknown(index_set(i))*PMFrare(index_set(i),tempR(j))...
                            *log(PMFunknown(index_set(i))*PMFrare(index_set(i),tempR(j)));
                    end
                end
                %add up the unknown unknowns
                if PMFrare(index_set(i),end)~=0
                    ent_temp = ent_temp-...
                        PMFunknown(index_set(i))*PMFrare(index_set(i),end)...
                        *log(PMFunknown(index_set(i))*PMFrare(index_set(i),end));
                end                
            else
                ent_temp = ent_temp-...
                    PMFunknown(index_set(i))*log(PMFunknown(index_set(i)));         
            end
            if length(tempN)>=2
                for j=1:length(tempN)
                    if PMFnormal(index_set(i),tempN(j))~=0
                        ent_temp = ent_temp-...
                            (1-PMFunknown(index_set(i)))*PMFnormal(index_set(i),tempN(j))*...
                            log((1-PMFunknown(index_set(i)))*PMFnormal(index_set(i),tempN(j)));
                    end
                end
            else
                ent_temp = ent_temp-...
                    (1-PMFunknown(index_set(i)))*log((1-PMFunknown(index_set(i))));         
            end
        else
            ent_temp = 0;
        end
        if ent_temp>max_ent
            max_ent = ent_temp;
            sample2label = index_set(i);
        end
    end
elseif what==3 %highest unknown category
    sus_temp = 0;
    for i=1:length(index_set)
        temp = PMFunknown(index_set(i))*PMFrare(index_set(i),end);
        if temp>sus_temp
            sus_temp = temp;
            sample2label = index_set(i);
        end
    end
        
elseif what==-1 %random sampling
    sample2label = index_set(randsample(length(index_set),1));
end
end