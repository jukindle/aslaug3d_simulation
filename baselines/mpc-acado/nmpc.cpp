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
    Control j1_p_eps;
    Control j2_p_eps;
    Control j1_v_eps;
    Control j2_v_eps;
    Control mb_v_x_eps;
    Control mb_v_y_eps;
    Control mb_v_th_eps;
    DifferentialState lx_1;
    DifferentialState ly_1;
    Control leps_1;
    Control lepsl1_1;
    Control lepsl2_1;
    Control lepsl3_1;
    IntermediateState intS1 = (-(-1.03999999999999995226e-01)*(-j1_v-j2_v-mb_v_th)-(-j1_v-mb_v_th)*(-sin(j2_p))*3.77000000000000001776e-01-(-j1_v-mb_v_th)*cos(j2_p)*7.39999999999999963363e-02-cos((j1_p+j2_p))*mb_v_x-mb_v_y*sin((j1_p+j2_p)));
    IntermediateState intS2 = (-(-sin((j1_p+j2_p)))*mb_v_x-(j1_v+j2_v+mb_v_th)*7.33000000000000095923e-01-(j1_v+mb_v_th)*cos(j2_p)*3.77000000000000001776e-01-(j1_v+mb_v_th)*sin(j2_p)*7.39999999999999963363e-02-cos((j1_p+j2_p))*mb_v_y);
    IntermediateState intS3 = ((-j1_v)*cos(j1_p)*7.39999999999999963363e-02+(-j1_v)*sin(j1_p)*3.77000000000000001776e-01);
    IntermediateState intS4 = ((-sin(j1_p))*j1_v*7.39999999999999963363e-02+3.77000000000000001776e-01*cos(j1_p)*j1_v);
    IntermediateState intS5 = ((-1.03999999999999995226e-01)*(-j2_v)*cos((j1_p+j2_p))+(-j2_v)*sin((j1_p+j2_p))*4.61000000000000020872e-01+intS3);
    IntermediateState intS6 = ((-1.03999999999999995226e-01)*(-sin((j1_p+j2_p)))*j2_v+4.61000000000000020872e-01*cos((j1_p+j2_p))*j2_v+intS4);
    IntermediateState intS7 = ((-1.03999999999999995226e-01)*(-j2_v)*cos((j1_p+j2_p))+(-j2_v)*sin((j1_p+j2_p))*7.33000000000000095923e-01+intS3);
    IntermediateState intS8 = ((-1.03999999999999995226e-01)*(-sin((j1_p+j2_p)))*j2_v+7.33000000000000095923e-01*cos((j1_p+j2_p))*j2_v+intS4);
    IntermediateState intS9 = (ly_1*mb_v_th-mb_v_x);
    IntermediateState intS10 = ((-mb_v_th)*lx_1-mb_v_y);
    Function acadodata_f2;
    acadodata_f2 << sp_x;
    Function acadodata_f3;
    acadodata_f3 << sp_y;
    Function acadodata_f4;
    acadodata_f4 << 1.00000000000000000000e+02*j1_p_eps;
    Function acadodata_f5;
    acadodata_f5 << 1.00000000000000000000e+02*j1_v_eps;
    Function acadodata_f6;
    acadodata_f6 << 1.00000000000000000000e+02*j2_p_eps;
    Function acadodata_f7;
    acadodata_f7 << 1.00000000000000000000e+02*j2_v_eps;
    Function acadodata_f8;
    acadodata_f8 << 1.00000000000000000000e+02*mb_v_x_eps;
    Function acadodata_f9;
    acadodata_f9 << 1.00000000000000000000e+02*mb_v_y_eps;
    Function acadodata_f10;
    acadodata_f10 << 1.00000000000000000000e+02*mb_v_th_eps;
    Function acadodata_f11;
    acadodata_f11 << 1.00000000000000000000e+02*leps_1;
    Function acadodata_f12;
    acadodata_f12 << 1.00000000000000000000e+02*lepsl1_1;
    Function acadodata_f13;
    acadodata_f13 << 1.00000000000000000000e+02*lepsl2_1;
    Function acadodata_f14;
    acadodata_f14 << 1.00000000000000000000e+02*lepsl3_1;
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
    acadodata_f1 << dot(l2_x) == intS3;
    acadodata_f1 << dot(l2_y) == intS4;
    acadodata_f1 << dot(l3_x) == intS3;
    acadodata_f1 << dot(l3_y) == intS4;
    acadodata_f1 << dot(lx_1) == intS9;
    acadodata_f1 << dot(ly_1) == intS10;

    OCP ocp1(0, 1, 4);
    ocp1.minimizeLSQ(acadodata_f2);
    ocp1.minimizeLSQ(acadodata_f3);
    ocp1.minimizeLSQ(acadodata_f4);
    ocp1.minimizeLSQ(acadodata_f5);
    ocp1.minimizeLSQ(acadodata_f6);
    ocp1.minimizeLSQ(acadodata_f7);
    ocp1.minimizeLSQ(acadodata_f8);
    ocp1.minimizeLSQ(acadodata_f9);
    ocp1.minimizeLSQ(acadodata_f10);
    ocp1.minimizeLSQ(acadodata_f11);
    ocp1.minimizeLSQ(acadodata_f12);
    ocp1.minimizeLSQ(acadodata_f13);
    ocp1.minimizeLSQ(acadodata_f14);
    ocp1.subjectTo(acadodata_f1);
    ocp1.subjectTo((-1.44999999999999995559e+00) <= u_x <= 1.44999999999999995559e+00);
    ocp1.subjectTo((-1.44999999999999995559e+00) <= u_y <= 1.44999999999999995559e+00);
    ocp1.subjectTo((-4.50000000000000011102e-01) <= u_th <= 4.50000000000000011102e-01);
    ocp1.subjectTo((-6.99999999999999955591e-01) <= u_j1 <= 6.99999999999999955591e-01);
    ocp1.subjectTo((-6.99999999999999955591e-01) <= u_j2 <= 6.99999999999999955591e-01);
    ocp1.subjectTo((-2.89000000000000012434e+00) <= (j1_p+j1_p_eps));
    ocp1.subjectTo((j1_p-j1_p_eps) <= 2.89000000000000012434e+00);
    ocp1.subjectTo(j1_p_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-1.00000000000000000000e+00) <= (j1_v+j1_v_eps));
    ocp1.subjectTo((j1_v-j1_v_eps) <= 1.00000000000000000000e+00);
    ocp1.subjectTo(j1_v_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo(5.00000000000000027756e-02 <= (j2_p+j2_p_eps));
    ocp1.subjectTo((j2_p-j2_p_eps) <= 3.00000000000000000000e+00);
    ocp1.subjectTo(j2_p_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-2.10000000000000000000e+01) <= (j2_v+j2_v_eps));
    ocp1.subjectTo((j2_v-j2_v_eps) <= 1.00000000000000000000e+00);
    ocp1.subjectTo(j2_v_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-3.49999999999999977796e-01) <= (mb_v_x+mb_v_x_eps));
    ocp1.subjectTo((mb_v_x-mb_v_x_eps) <= 3.49999999999999977796e-01);
    ocp1.subjectTo(mb_v_x_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-3.49999999999999977796e-01) <= (mb_v_y+mb_v_y_eps));
    ocp1.subjectTo((mb_v_y-mb_v_y_eps) <= 3.49999999999999977796e-01);
    ocp1.subjectTo(mb_v_y_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo((-6.99999999999999955591e-01) <= (mb_v_th+mb_v_th_eps));
    ocp1.subjectTo((mb_v_th-mb_v_th_eps) <= 6.99999999999999955591e-01);
    ocp1.subjectTo(mb_v_th_eps >= 0.00000000000000000000e+00);
    ocp1.subjectTo(3.49999999999999977796e-01 <= (leps_1+pow(lx_1,2.00000000000000000000e+00)+pow(ly_1,2.00000000000000000000e+00)));
    ocp1.subjectTo(leps_1 >= 0.00000000000000000000e+00);
    ocp1.subjectTo(1.49999999999999994449e-01 <= (lepsl1_1+pow((-l1_x+lx_1),2.00000000000000000000e+00)+pow((-l1_y+ly_1),2.00000000000000000000e+00)));
    ocp1.subjectTo(lepsl1_1 >= 0.00000000000000000000e+00);
    ocp1.subjectTo(1.49999999999999994449e-01 <= (lepsl2_1+pow((-l2_x+lx_1),2.00000000000000000000e+00)+pow((-l2_y+ly_1),2.00000000000000000000e+00)));
    ocp1.subjectTo(lepsl2_1 >= 0.00000000000000000000e+00);
    ocp1.subjectTo(5.00000000000000027756e-02 <= (lepsl3_1+pow((-l3_x+lx_1),2.00000000000000000000e+00)+pow((-l3_y+ly_1),2.00000000000000000000e+00)));
    ocp1.subjectTo(lepsl3_1 >= 0.00000000000000000000e+00);


    RealTimeAlgorithm algo1(ocp1, 0.2);
    algo1.set( MAX_NUM_ITERATIONS, 3 );
    algo1.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
    algo1.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
    algo1.set( MAX_NUM_QP_ITERATIONS, 500 );
    algo1.set( HOTSTART_QP, YES );
    algo1.set( LEVENBERG_MARQUARDT, 1.000000E-10 );

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

