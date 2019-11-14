/*
*    This file is part of ACADO Toolkit.
*
*    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
*    Copyright (C) 2008-2009 by Boris Houska and Hans Joachim Ferreau, K.U.Leuven.
*    Developed within the Optimization in Engineering Center (OPTEC) under
*    supervision of Moritz Diehl. All rights reserved.
*
*    ACADO Toolkit is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 3 of the License, or (at your option) any later version.
*
*    ACADO Toolkit is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with ACADO Toolkit; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*
*/


/**
*    Author David Ariens, Rien Quirynen
*    Date 2009-2013
*    http://www.acadotoolkit.org/matlab 
*/

#include <acado_optimal_control.hpp>
#include <acado_toolkit.hpp>
#include <acado/utils/matlab_acado_utils.hpp>

USING_NAMESPACE_ACADO

#include <mex.h>


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) 
 { 
 
    MatlabConsoleStreamBuf mybuf;
    RedirectStream redirect(std::cout, mybuf);
    clearAllStaticCounters( ); 
 
    mexPrintf("\nACADO Toolkit for Matlab - Developed by David Ariens and Rien Quirynen, 2009-2013 \n"); 
    mexPrintf("Support available at http://www.acadotoolkit.org/matlab \n \n"); 

    if (nrhs != 1){ 
      mexErrMsgTxt("This problem expects 1 right hand side argument(s) since you have defined 1 MexInput(s)");
    } 
 
    TIME autotime;
    int mexinput0_count = 0;
    if (mxGetM(prhs[0]) == 1 && mxGetN(prhs[0]) >= 1) 
       mexinput0_count = mxGetN(prhs[0]);
    else if (mxGetM(prhs[0]) >= 1 && mxGetN(prhs[0]) == 1) 
       mexinput0_count = mxGetM(prhs[0]);
    else 
       mexErrMsgTxt("Input 0 must be a noncomplex double vector of dimension 1xY.");

    double *mexinput0_temp = NULL; 
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) { 
      mexErrMsgTxt("Input 0 must be a noncomplex double vector of dimension 1xY.");
    } 
    mexinput0_temp = mxGetPr(prhs[0]); 
    DVector mexinput0(mexinput0_count);
    for( int i=0; i<mexinput0_count; ++i ){ 
        mexinput0(i) = mexinput0_temp[i];
    } 

    Control u_x;
    Control u_y;
    Control u_th;
    Control u_j1;
    Control u_j2;
    DifferentialState sp_x;
    DifferentialState sp_y;
    DifferentialState j1_p;
    DifferentialState j2_p;
    DifferentialState j1_v;
    DifferentialState j2_v;
    DifferentialState mb_v_x;
    DifferentialState mb_v_y;
    DifferentialState mb_v_th;
    DifferentialState l1_x;
    DifferentialState l1_y;
    DifferentialState l2_x;
    DifferentialState l2_y;
    DifferentialState l3_x;
    DifferentialState l3_y;
    DifferentialState w1x;
    DifferentialState w1y;
    DifferentialState w2x;
    DifferentialState w2y;
    Control j_p_eps;
    Control j_v_eps;
    Control mb_v_eps;
    Control bw_eps;
    Control lw_eps;
    IntermediateState intS1 = (-(-8.30000000000000043299e-02)*(-j1_v-j2_v-mb_v_th)-(-j1_v-mb_v_th)*(-sin(j2_p))*3.16000000000000003109e-01-(-j1_v-mb_v_th)*cos(j2_p)*8.20000000000000034417e-02-cos((j1_p+j2_p))*mb_v_x-mb_v_y*sin((j1_p+j2_p)));
    IntermediateState intS2 = (-(-sin((j1_p+j2_p)))*mb_v_x-(j1_v+j2_v+mb_v_th)*5.94999999999999973355e-01-(j1_v+mb_v_th)*cos(j2_p)*3.16000000000000003109e-01-(j1_v+mb_v_th)*sin(j2_p)*8.20000000000000034417e-02-cos((j1_p+j2_p))*mb_v_y);
    IntermediateState intS3 = ((-j1_v)*cos(j1_p)*8.20000000000000034417e-02+(-j1_v)*sin(j1_p)*3.16000000000000003109e-01);
    IntermediateState intS4 = ((-sin(j1_p))*j1_v*8.20000000000000034417e-02+3.16000000000000003109e-01*cos(j1_p)*j1_v);
    IntermediateState intS5 = ((-8.30000000000000043299e-02)*(-j2_v)*cos((j1_p+j2_p))+(-j2_v)*sin((j1_p+j2_p))*3.84000000000000007994e-01+intS3);
    IntermediateState intS6 = ((-8.30000000000000043299e-02)*(-sin((j1_p+j2_p)))*j2_v+3.84000000000000007994e-01*cos((j1_p+j2_p))*j2_v+intS4);
    IntermediateState intS7 = ((-8.30000000000000043299e-02)*(-j2_v)*cos((j1_p+j2_p))+(-j2_v)*sin((j1_p+j2_p))*5.94999999999999973355e-01+intS3);
    IntermediateState intS8 = ((-8.30000000000000043299e-02)*(-sin((j1_p+j2_p)))*j2_v+5.94999999999999973355e-01*cos((j1_p+j2_p))*j2_v+intS4);
    IntermediateState intS9 = (mb_v_th*w1y-mb_v_x);
    IntermediateState intS10 = ((-mb_v_th)*w1x-mb_v_y);
    IntermediateState intS11 = (mb_v_th*w2y-mb_v_x);
    IntermediateState intS12 = ((-mb_v_th)*w2x-mb_v_y);
    IntermediateState intS13 = (pow((2.99999999999999988898e-01+w1x),2.00000000000000000000e+00)+pow(w1y,2.00000000000000000000e+00));
    IntermediateState intS14 = (pow((2.99999999999999988898e-01+w2x),2.00000000000000000000e+00)+pow(w2y,2.00000000000000000000e+00));
    IntermediateState intS15 = (l1_x*w1y-l1_y*w1x)/(pow(w1x,2.00000000000000000000e+00)+pow(w1y,2.00000000000000000000e+00));
    IntermediateState intS16 = (intS15*w1y+w1x);
    IntermediateState intS17 = (-intS15*w1x+w1y);
    IntermediateState intS18 = (pow((intS16-l1_x),2.00000000000000000000e+00)+pow((intS17-l1_y),2.00000000000000000000e+00));
    IntermediateState intS19 = (l1_x*w2y-l1_y*w2x)/(pow(w2x,2.00000000000000000000e+00)+pow(w2y,2.00000000000000000000e+00));
    IntermediateState intS20 = (intS19*w2y+w2x);
    IntermediateState intS21 = (-intS19*w2x+w2y);
    IntermediateState intS22 = (pow((intS20-l1_x),2.00000000000000000000e+00)+pow((intS21-l1_y),2.00000000000000000000e+00));
    IntermediateState intS23 = (l2_x*w1y-l2_y*w1x)/(pow(w1x,2.00000000000000000000e+00)+pow(w1y,2.00000000000000000000e+00));
    IntermediateState intS24 = (intS23*w1y+w1x);
    IntermediateState intS25 = (-intS23*w1x+w1y);
    IntermediateState intS26 = (pow((intS24-l2_x),2.00000000000000000000e+00)+pow((intS25-l2_y),2.00000000000000000000e+00));
    IntermediateState intS27 = (l2_x*w2y-l2_y*w2x)/(pow(w2x,2.00000000000000000000e+00)+pow(w2y,2.00000000000000000000e+00));
    IntermediateState intS28 = (intS27*w2y+w2x);
    IntermediateState intS29 = (-intS27*w2x+w2y);
    IntermediateState intS30 = (pow((intS28-l2_x),2.00000000000000000000e+00)+pow((intS29-l2_y),2.00000000000000000000e+00));
    IntermediateState intS31 = (l3_x*w1y-l3_y*w1x)/(pow(w1x,2.00000000000000000000e+00)+pow(w1y,2.00000000000000000000e+00));
    IntermediateState intS32 = (intS31*w1y+w1x);
    IntermediateState intS33 = (-intS31*w1x+w1y);
    IntermediateState intS34 = (pow((intS32-l3_x),2.00000000000000000000e+00)+pow((intS33-l3_y),2.00000000000000000000e+00));
    IntermediateState intS35 = (l3_x*w2y-l3_y*w2x)/(pow(w2x,2.00000000000000000000e+00)+pow(w2y,2.00000000000000000000e+00));
    IntermediateState intS36 = (intS35*w2y+w2x);
    IntermediateState intS37 = (-intS35*w2x+w2y);
    IntermediateState intS38 = (pow((intS36-l3_x),2.00000000000000000000e+00)+pow((intS37-l3_y),2.00000000000000000000e+00));
    Function acadodata_f2;
    acadodata_f2 << 1.00000000000000000000e+04*sp_x;
    DVector acadodata_v1(1);
    acadodata_v1(0) = 0;
    Function acadodata_f3;
    acadodata_f3 << 1.00000000000000000000e+04*sp_y;
    DVector acadodata_v2(1);
    acadodata_v2(0) = 0;
    Function acadodata_f4;
    acadodata_f4 << 1.00000000000000000000e+01*mb_v_x;
    DVector acadodata_v3(1);
    acadodata_v3(0) = 0;
    Function acadodata_f5;
    acadodata_f5 << 1.00000000000000000000e+01*mb_v_y;
    DVector acadodata_v4(1);
    acadodata_v4(0) = 0;
    Function acadodata_f6;
    acadodata_f6 << 5.00000000000000000000e+00*j1_v;
    DVector acadodata_v5(1);
    acadodata_v5(0) = 0;
    Function acadodata_f7;
    acadodata_f7 << 5.00000000000000000000e+00*j2_v;
    DVector acadodata_v6(1);
    acadodata_v6(0) = 0;
    Function acadodata_f8;
    acadodata_f8 << 1.00000000000000000000e+01*mb_v_eps;
    Function acadodata_f9;
    acadodata_f9 << 1.00000000000000000000e+03*j_p_eps;
    Function acadodata_f10;
    acadodata_f10 << 1.00000000000000000000e+01*j_v_eps;
    Function acadodata_f11;
    acadodata_f11 << 1.00000000000000000000e+03*bw_eps;
    Function acadodata_f12;
    acadodata_f12 << 1.00000000000000000000e+03*lw_eps;
    DifferentialEquation acadodata_f1;
    acadodata_f1 << dot(mb_v_x) == u_x;
    acadodata_f1 << dot(mb_v_y) == u_y;
    acadodata_f1 << dot(mb_v_th) == u_th;
    acadodata_f1 << dot(j1_p) == j1_v;
    acadodata_f1 << dot(j2_p) == j2_v;
    acadodata_f1 << dot(j1_v) == u_j1;
    acadodata_f1 << dot(j2_v) == u_j2;
    acadodata_f1 << dot(sp_x) == intS1;
    acadodata_f1 << dot(sp_y) == intS2;
    acadodata_f1 << dot(l1_x) == intS3;
    acadodata_f1 << dot(l1_y) == intS4;
    acadodata_f1 << dot(l2_x) == intS5;
    acadodata_f1 << dot(l2_y) == intS6;
    acadodata_f1 << dot(l3_x) == intS7;
    acadodata_f1 << dot(l3_y) == intS8;
    acadodata_f1 << dot(w1x) == intS9;
    acadodata_f1 << dot(w1y) == intS10;
    acadodata_f1 << dot(w2x) == intS11;
    acadodata_f1 << dot(w2y) == intS12;

    OCP ocp1(0, 3, 10);
    ocp1.minimizeLSQ(acadodata_f8);
    ocp1.minimizeLSQ(acadodata_f9);
    ocp1.minimizeLSQ(acadodata_f10);
    ocp1.minimizeLSQ(acadodata_f11);
    ocp1.minimizeLSQ(acadodata_f12);
    ocp1.minimizeLSQEndTerm(acadodata_f2, acadodata_v1);
    ocp1.minimizeLSQEndTerm(acadodata_f3, acadodata_v2);
    ocp1.minimizeLSQEndTerm(acadodata_f4, acadodata_v3);
    ocp1.minimizeLSQEndTerm(acadodata_f5, acadodata_v4);
    ocp1.minimizeLSQEndTerm(acadodata_f6, acadodata_v5);
    ocp1.minimizeLSQEndTerm(acadodata_f7, acadodata_v6);
    ocp1.subjectTo(acadodata_f1);
    ocp1.subjectTo((-1.44999999999999995559e+00) <= u_x <= 1.44999999999999995559e+00);
    ocp1.subjectTo((-1.44999999999999995559e+00) <= u_y <= 1.44999999999999995559e+00);
    ocp1.subjectTo((-4.50000000000000011102e-01) <= u_th <= 4.50000000000000011102e-01);
    ocp1.subjectTo((-6.99999999999999955591e-01) <= u_j1 <= 6.99999999999999955591e-01);
    ocp1.subjectTo((-6.99999999999999955591e-01) <= u_j2 <= 6.99999999999999955591e-01);
    ocp1.subjectTo((-2.89000000000000012434e+00) <= (j1_p+j_p_eps));
    ocp1.subjectTo((j1_p-j_p_eps) <= 2.89000000000000012434e+00);
    ocp1.subjectTo((-1.00000000000000000000e+00) <= (j1_v+j_v_eps));
    ocp1.subjectTo((j1_v-j_v_eps) <= 1.00000000000000000000e+00);
    ocp1.subjectTo(5.00000000000000027756e-02 <= (j2_p+j_p_eps));
    ocp1.subjectTo((j2_p-j_p_eps) <= 2.50000000000000000000e+00);
    ocp1.subjectTo((-2.10000000000000000000e+01) <= (j2_v+j_v_eps));
    ocp1.subjectTo((j2_v-j_v_eps) <= 1.00000000000000000000e+00);
    ocp1.subjectTo((-3.49999999999999977796e-01) <= (mb_v_eps+mb_v_x));
    ocp1.subjectTo((-mb_v_eps+mb_v_x) <= 3.49999999999999977796e-01);
    ocp1.subjectTo((-3.49999999999999977796e-01) <= (mb_v_eps+mb_v_y));
    ocp1.subjectTo((-mb_v_eps+mb_v_y) <= 3.49999999999999977796e-01);
    ocp1.subjectTo((-6.99999999999999955591e-01) <= (mb_v_eps+mb_v_th));
    ocp1.subjectTo((-mb_v_eps+mb_v_th) <= 6.99999999999999955591e-01);
    ocp1.subjectTo(3.59999999999999986677e-01 <= (bw_eps+intS13));
    ocp1.subjectTo(3.59999999999999986677e-01 <= (bw_eps+intS14));
    ocp1.subjectTo(6.25000000000000000000e-02 <= (intS18+lw_eps));
    ocp1.subjectTo(6.25000000000000000000e-02 <= (intS22+lw_eps));
    ocp1.subjectTo(2.24999999999999991673e-02 <= (intS26+lw_eps));
    ocp1.subjectTo(2.24999999999999991673e-02 <= (intS30+lw_eps));
    ocp1.subjectTo(9.99999999999999954748e-07 <= (intS34+lw_eps));
    ocp1.subjectTo(9.99999999999999954748e-07 <= (intS38+lw_eps));
    ocp1.subjectTo(mb_v_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo(j_p_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo(j_v_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo(bw_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo(lw_eps >= 0.00000000000000000000e+00);


    RealTimeAlgorithm algo1(ocp1, 0.2);

    Controller controller1( algo1 );
    controller1.init(0, mexinput0);
    controller1.step(0, mexinput0);

    const char* outputFieldNames[] = {"U", "P"}; 
    plhs[0] = mxCreateStructMatrix( 1,1,2,outputFieldNames ); 
    mxArray *OutU = NULL;
    double  *outU = NULL;
    OutU = mxCreateDoubleMatrix( 1,controller1.getNU(),mxREAL ); 
    outU = mxGetPr( OutU );
    DVector vec_outU; 
    controller1.getU(vec_outU); 
    for( int i=0; i<vec_outU.getDim(); ++i ){ 
        outU[i] = vec_outU(i); 
    } 

    mxArray *OutP = NULL;
    double  *outP = NULL;
    OutP = mxCreateDoubleMatrix( 1,controller1.getNP(),mxREAL ); 
    outP = mxGetPr( OutP );
    DVector vec_outP; 
    controller1.getP(vec_outP); 
    for( int i=0; i<vec_outP.getDim(); ++i ){ 
        outP[i] = vec_outP(i); 
    } 

    mxSetField( plhs[0],0,"U",OutU );
    mxSetField( plhs[0],0,"P",OutP );


    clearAllStaticCounters( ); 
 
} 

