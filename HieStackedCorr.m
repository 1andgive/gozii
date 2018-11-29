function [Umat_list,V_result] = HieStackedCorr(Vmat, num_stages)


num_objects=size(Vmat,2);
% Umat=zeros(num_objects,num_objects);
Umat_list=cell(1,num_stages);
for i =1:num_stages
    psi=Vmat'*Vmat;
    Inverse_diag=(1./diag(psi)).^0.5;
    Corr=Inverse_diag*Inverse_diag'.*psi;
    Umat=1-Corr;
    Umat_list{i}=Umat;
    Vmat=(Umat*Vmat'+Vmat')';
end
V_result=Vmat;