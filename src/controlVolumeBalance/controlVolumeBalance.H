/*---------------------------------------------------------------------------*\
Class
    Foam::functionObjects::controlVolumeBalance

Description
    Integral mass and momentum balance of control volumes next to a flat
    wall, from the face fluxes of the finite-volume discretisation: what the
    solver itself exchanges across the control-volume boundary.

    A control volume is the set of cells with their centroid below a height h
    above the wall, cut into streamwise bins.  Its boundary faces are grouped
    into

        top         faces to a cell above h in the same bin
        upstream    faces to a cell of another bin (or outside the x range)
        downstream  with a smaller / larger streamwise coordinate
        <patch>     boundary faces, one group per non-empty patch

    and for every group the sums over its faces are written, all as a gain of
    the control volume (S_f pointing out of it):

        massIn      -sum phi_f                                      [kg/s]
        convection  -sum phi_f (U_f - URef), U_f with the div(phi,U) scheme
        convectionLinear   the same with linearly interpolated U_f; the
                    difference is the numerical flux of the convection scheme
        pressure    -sum (p_f - pRef) S_f
        viscous     sum of the laminar stress flux
        reynolds    sum of the Reynolds-stress flux (Reynolds-stress model:
                    -(rho R)_f . S_f; eddy-viscosity model: the nut part of
                    the stress flux)                                 [N]

    The stress fluxes repeat rhoSimpleFoam's UEqn face by face (same
    expressions as the turbulenceBudget function object, same fvSchemes), so
    for a converged steady solution the momentum terms of all groups of one
    control volume add up to zero, and so does massIn.  With URef the free
    stream velocity the convection is the flux of the momentum deficit.

    With `interpolate true` a cell that straddles h belongs to the control
    volume with the fraction of its wall-normal extent below h, and a face
    counts with the difference of the two fractions: on a structured mesh
    this interpolates every term linearly between the two cell rows around h.

Usage
    Run with the solver in post-processing mode:

        cvBalance
        {
            type        controlVolumeBalance;
            libs        ("<path>/libcontrolVolumeBalance.so");
            heights     (1.3e-4 3.1e-3);        // h, wall-normal coordinate
            edges       (0 0.05 0.1 0.15);      // streamwise bin edges
            interpolate false;                  // optional
            URef        (69.4 0 0);             // optional, default 0
            pRef        110000;                 // optional, default 0
            origin      (0 0 0);                // optional
            streamwise  (1 0 0);                // optional
            wallNormal  (0 1 0);                // optional
        }

    rhoSimpleFoam -postProcess -latestTime -dict <file>

    Output: postProcessing/<name>/<time>/balance.dat, one row per height,
    bin and group with faces: the face area, its area-weighted mean position
    (streamwise, wallNormal), the outward area vector and the sums above,
    vectors as (streamwise, wallNormal) components.

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_controlVolumeBalance_H
#define functionObjects_controlVolumeBalance_H

#include "fvMeshFunctionObject.H"
#include "writeFile.H"
#include "volFields.H"
#include "surfaceFields.H"

namespace Foam
{
namespace functionObjects
{

class controlVolumeBalance
:
    public fvMeshFunctionObject,
    public writeFile
{
    // Private Data

        scalarList heights_;

        scalarList edges_;

        bool interpolate_;

        vector URef_;

        scalar pRef_;

        point origin_;

        vector streamwise_;

        vector wallNormal_;


    // Private Member Functions

        ITstream& gaussScheme(ITstream& is, const word& name) const;

        template<class Type>
        tmp<surfaceVectorField> divFlux
        (
            const GeometricField<Type, fvPatchField, volMesh>& vf,
            const word& name
        ) const;

        tmp<surfaceVectorField> laplacianFlux
        (
            const volScalarField& gamma,
            const volVectorField& U
        ) const;

        tmp<surfaceScalarField> pressureFace(const volScalarField& p) const;

        void cellFractions
        (
            const scalarField& yCell,
            List<scalarField>& fraction
        ) const;


public:

    TypeName("controlVolumeBalance");


    // Constructors

        controlVolumeBalance
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    virtual ~controlVolumeBalance() = default;


    // Member Functions

        virtual bool read(const dictionary& dict);

        virtual bool execute();

        virtual bool write();
};


} // End namespace functionObjects
} // End namespace Foam

#endif
