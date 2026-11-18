#include "controlVolumeBalance.H"
#include "turbulentFluidThermoModel.H"
#include "fvc.H"
#include "snGradScheme.H"
#include "surfaceInterpolationScheme.H"
#include "syncTools.H"
#include "emptyPolyPatch.H"
#include "addToRunTimeSelectionTable.H"
#include <algorithm>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(controlVolumeBalance, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        controlVolumeBalance,
        dictionary
    );
}
}

namespace
{
    // Groups of control-volume faces that are not boundary patches
    enum { TOP, UPSTREAM, DOWNSTREAM, N_INTERNAL_GROUPS };

    // Momentum face fluxes
    enum { CONVECTION, CONVECTION_LINEAR, PRESSURE, VISCOUS, REYNOLDS, N_FLUX };

    // Sums per height, bin and group
    enum { AREA, AREA_S, AREA_N, S_S, S_N, MASS, FLUX0 };
    const Foam::label nSums = FLUX0 + 2*N_FLUX;
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::ITstream& Foam::functionObjects::controlVolumeBalance::gaussScheme
(
    ITstream& is,
    const word& name
) const
{
    const word type(is);
    if (type != "Gauss")
    {
        FatalErrorInFunction
            << "Scheme " << name << " is " << type << ": only Gauss schemes"
            << " are a sum of face fluxes" << exit(FatalError);
    }
    return is;
}


template<class Type>
Foam::tmp<Foam::surfaceVectorField>
Foam::functionObjects::controlVolumeBalance::divFlux
(
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
) const
{
    ITstream& is = gaussScheme(mesh_.divScheme(name), name);
    return
        surfaceInterpolationScheme<Type>::New(mesh_, is)().dotInterpolate
        (
            mesh_.Sf(),
            vf
        );
}


Foam::tmp<Foam::surfaceVectorField>
Foam::functionObjects::controlVolumeBalance::laplacianFlux
(
    const volScalarField& gamma,
    const volVectorField& U
) const
{
    const word name("laplacian(" + gamma.name() + ',' + U.name() + ')');
    ITstream& is = gaussScheme(mesh_.laplacianScheme(name), name);
    tmp<surfaceInterpolationScheme<scalar>> tinterp
    (
        surfaceInterpolationScheme<scalar>::New(mesh_, is)
    );
    tmp<fv::snGradScheme<vector>> tsnGrad
    (
        fv::snGradScheme<vector>::New(mesh_, is)
    );
    return tinterp().interpolate(gamma)*mesh_.magSf()*tsnGrad().snGrad(U);
}


Foam::tmp<Foam::surfaceScalarField>
Foam::functionObjects::controlVolumeBalance::pressureFace
(
    const volScalarField& p
) const
{
    const word name("grad(" + p.name() + ')');
    ITstream& is = gaussScheme(mesh_.gradScheme(name), name);
    if (is.eof())
    {
        return linearInterpolate(p);
    }
    return surfaceInterpolationScheme<scalar>::New(mesh_, is)().interpolate(p);
}


void Foam::functionObjects::controlVolumeBalance::cellFractions
(
    const scalarField& yCell,
    List<scalarField>& fraction
) const
{
    fraction.resize(heights_.size());

    scalarField yLow;
    scalarField yHigh;
    if (interpolate_)
    {
        yLow.resize(mesh_.nCells(), GREAT);
        yHigh.resize(mesh_.nCells(), -GREAT);
        const faceList& faces = mesh_.faces();
        const pointField& points = mesh_.points();
        forAll(mesh_.cells(), celli)
        {
            for (const label facei : mesh_.cells()[celli])
            {
                for (const label pointi : faces[facei])
                {
                    const scalar y = (points[pointi] - origin_) & wallNormal_;
                    yLow[celli] = min(yLow[celli], y);
                    yHigh[celli] = max(yHigh[celli], y);
                }
            }
        }
    }

    forAll(heights_, k)
    {
        const scalar h = heights_[k];
        scalarField& f = fraction[k];
        f.resize(mesh_.nCells());
        forAll(f, celli)
        {
            if (interpolate_)
            {
                f[celli] =
                    min
                    (
                        max
                        (
                            (h - yLow[celli])
                           /max(yHigh[celli] - yLow[celli], VSMALL),
                            scalar(0)
                        ),
                        scalar(1)
                    );
            }
            else
            {
                f[celli] = yCell[celli] < h ? 1 : 0;
            }
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::controlVolumeBalance::controlVolumeBalance
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    writeFile(mesh_, name, typeName, dict)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::controlVolumeBalance::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);
    writeFile::read(dict);

    dict.readEntry("heights", heights_);
    dict.readEntry("edges", edges_);
    interpolate_ = dict.getOrDefault<bool>("interpolate", false);
    URef_ = dict.getOrDefault<vector>("URef", Zero);
    pRef_ = dict.getOrDefault<scalar>("pRef", 0);
    origin_ = dict.getOrDefault<point>("origin", Zero);
    streamwise_ = normalised(dict.getOrDefault<vector>("streamwise", vector(1, 0, 0)));
    wallNormal_ = normalised(dict.getOrDefault<vector>("wallNormal", vector(0, 1, 0)));

    if (edges_.size() < 2)
    {
        FatalIOErrorInFunction(dict)
            << "edges needs at least two values" << exit(FatalIOError);
    }
    for (label i = 1; i < edges_.size(); ++i)
    {
        if (edges_[i] <= edges_[i - 1])
        {
            FatalIOErrorInFunction(dict)
                << "edges must increase" << exit(FatalIOError);
        }
    }

    return true;
}


bool Foam::functionObjects::controlVolumeBalance::execute()
{
    return true;
}


bool Foam::functionObjects::controlVolumeBalance::write()
{
    if (!foundObject<compressible::turbulenceModel>(turbulenceModel::propertiesName))
    {
        WarningInFunction
            << "No compressible turbulence model - no balance" << endl;
        return false;
    }
    const auto& turb = lookupObject<compressible::turbulenceModel>
    (
        turbulenceModel::propertiesName
    );

    const volScalarField& rho = turb.rho();
    const surfaceScalarField& phi = turb.alphaRhoPhi();
    const volVectorField& U = turb.U();
    const volScalarField& p = lookupObject<volScalarField>("p");
    const tmp<volScalarField> tnu(turb.nu());
    const volScalarField& nu = tnu();
    const tmp<volScalarField> tnut(turb.nut());
    const volScalarField& nut = tnut();

    // Face fluxes of rhoSimpleFoam's UEqn as a gain of the face's owner,
    // term by term as turbulenceBudget::budgetMomentum
    const volTensorField gradU(fvc::grad(U));
    const volTensorField devGradUT(dev2(T(gradU)));
    const dimensionedVector URef("URef", dimVelocity, URef_);
    const dimensionedScalar pRef("pRef", p.dimensions(), pRef_);

    PtrList<surfaceVectorField> flux(N_FLUX);
    flux.set
    (
        CONVECTION,
        new surfaceVectorField
        (
            "convection",
            phi*URef - fvc::flux(phi, U, "div(phi,U)")
        )
    );
    flux.set
    (
        CONVECTION_LINEAR,
        new surfaceVectorField
        (
            "convectionLinear",
            phi*(URef - linearInterpolate(U))
        )
    );
    flux.set
    (
        PRESSURE,
        new surfaceVectorField
        (
            "pressure",
            (pRef - pressureFace(p))*mesh_.Sf()
        )
    );
    {
        const volScalarField mu(rho*nu);
        flux.set
        (
            VISCOUS,
            new surfaceVectorField
            (
                "viscous",
                laplacianFlux(mu, U)
              + divFlux
                (
                    volTensorField(mu*devGradUT),
                    "div(((rho*(thermo:mu|rho))*dev2(T(grad(U)))))"
                )
            )
        );
    }
    if (foundObject<volSymmTensorField>("R"))
    {
        const volSymmTensorField& R = lookupObject<volSymmTensorField>("R");
        const scalar couplingFactor =
            turb.subDict("RAS").optionalSubDict(turb.type() + "Coeffs")
           .getOrDefault<scalar>("couplingFactor", 0);

        flux.set
        (
            REYNOLDS,
            new surfaceVectorField
            (
                "reynolds",
                -divFlux(volSymmTensorField(rho*R), "div((rho*R))")
            )
        );
        if (couplingFactor > 0)
        {
            const volScalarField mut("nuEff", rho*nut);
            flux[REYNOLDS] -=
                couplingFactor
               *(
                    divFlux(volTensorField(mut*gradU), "div(devRhoReff)")
                  - laplacianFlux(mut, U)
                );
        }
    }
    else
    {
        const volScalarField mut(rho*nut);
        flux.set
        (
            REYNOLDS,
            new surfaceVectorField
            (
                "reynolds",
                laplacianFlux(mut, U)
              + divFlux
                (
                    volTensorField(mut*devGradUT),
                    "div(((rho*nuEff)*dev2(T(grad(U)))))"
                )
            )
        );
    }

    // Cells: streamwise bin and the fraction below each height
    const label nBins = edges_.size() - 1;
    const label nHeights = heights_.size();
    const vectorField& C = mesh_.cellCentres();
    scalarField xCell(mesh_.nCells());
    scalarField yCell(mesh_.nCells());
    labelList bin(mesh_.nCells(), -1);
    forAll(C, celli)
    {
        xCell[celli] = (C[celli] - origin_) & streamwise_;
        yCell[celli] = (C[celli] - origin_) & wallNormal_;
        if (xCell[celli] >= edges_.first() && xCell[celli] < edges_.last())
        {
            bin[celli] =
                std::upper_bound(edges_.begin(), edges_.end(), xCell[celli])
              - edges_.begin() - 1;
        }
    }
    List<scalarField> fraction;
    cellFractions(yCell, fraction);

    // The same of the cell behind every boundary face (coupled patches)
    const labelList nbrBin(syncTools::swapBoundaryCellList(mesh_, bin));
    const scalarList nbrX(syncTools::swapBoundaryCellList(mesh_, xCell));
    List<scalarList> nbrFraction(nHeights);
    forAll(fraction, k)
    {
        nbrFraction[k] = syncTools::swapBoundaryCellList(mesh_, fraction[k]);
    }

    // Groups: top, upstream, downstream, then the non-empty physical patches
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    wordList groupNames({"top", "upstream", "downstream"});
    labelList patchGroup(patches.size(), -1);
    forAll(patches, patchi)
    {
        if (!patches[patchi].coupled() && !isA<emptyPolyPatch>(patches[patchi]))
        {
            patchGroup[patchi] = groupNames.size();
            groupNames.append(patches[patchi].name());
        }
    }
    const label nGroups = groupNames.size();

    scalarField sums(nHeights*nBins*nGroups*nSums, Zero);

    // Add one face, with the weight and orientation of the cell it leaves
    auto add = [&]
    (
        const label k,
        const label b,
        const label g,
        const scalar weight,
        const vector& Sf,
        const point& Cf,
        const scalar phif,
        const FixedList<vector, N_FLUX>& fluxf
    )
    {
        scalar* s = &sums[((k*nBins + b)*nGroups + g)*nSums];
        const scalar area = mag(weight)*mag(Sf);
        s[AREA] += area;
        s[AREA_S] += area*((Cf - origin_) & streamwise_);
        s[AREA_N] += area*((Cf - origin_) & wallNormal_);
        s[S_S] += weight*(Sf & streamwise_);
        s[S_N] += weight*(Sf & wallNormal_);
        s[MASS] -= weight*phif;
        for (label i = 0; i < N_FLUX; ++i)
        {
            s[FLUX0 + 2*i] += weight*(fluxf[i] & streamwise_);
            s[FLUX0 + 2*i + 1] += weight*(fluxf[i] & wallNormal_);
        }
    };

    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();
    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceVectorField& Cf = mesh_.Cf();
    FixedList<vector, N_FLUX> fluxf;

    // The part of a face that bounds the control volume of cell `own`
    // towards the cell `nei`: top where nei lies above it, upstream or
    // downstream where nei belongs to another bin.  sign = -1 for the
    // neighbour side of an internal face (S_f points into the cell).
    auto addSide = [&]
    (
        const label k,
        const label own,
        const scalar fNei,
        const label binNei,
        const scalar xNei,
        const scalar sign,
        const vector& Sff,
        const point& Cff,
        const scalar phif
    )
    {
        const scalar fOwn = fraction[k][own];
        if (fOwn > fNei)
        {
            add(k, bin[own], TOP, sign*(fOwn - fNei), Sff, Cff, phif, fluxf);
        }
        if (binNei != bin[own] && min(fOwn, fNei) > 0)
        {
            add
            (
                k, bin[own], xNei > xCell[own] ? DOWNSTREAM : UPSTREAM,
                sign*min(fOwn, fNei), Sff, Cff, phif, fluxf
            );
        }
    };

    forAll(neighbour, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];
        if (bin[own] < 0 && bin[nei] < 0)
        {
            continue;
        }
        for (label i = 0; i < N_FLUX; ++i)
        {
            fluxf[i] = flux[i][facei];
        }
        forAll(heights_, k)
        {
            if (bin[own] >= 0)
            {
                addSide
                (
                    k, own, fraction[k][nei], bin[nei], xCell[nei], 1,
                    Sf[facei], Cf[facei], phi[facei]
                );
            }
            if (bin[nei] >= 0)
            {
                addSide
                (
                    k, nei, fraction[k][own], bin[own], xCell[own], -1,
                    Sf[facei], Cf[facei], phi[facei]
                );
            }
        }
    }

    forAll(patches, patchi)
    {
        const polyPatch& pp = patches[patchi];
        if (isA<emptyPolyPatch>(pp))
        {
            continue;
        }
        const labelUList& faceCells = pp.faceCells();
        const vectorField& pSf = Sf.boundaryField()[patchi];
        const vectorField& pCf = Cf.boundaryField()[patchi];
        const scalarField& pPhi = phi.boundaryField()[patchi];
        forAll(faceCells, i)
        {
            const label own = faceCells[i];
            if (bin[own] < 0)
            {
                continue;
            }
            const label bFacei = pp.start() - mesh_.nInternalFaces() + i;
            for (label j = 0; j < N_FLUX; ++j)
            {
                fluxf[j] = flux[j].boundaryField()[patchi][i];
            }
            forAll(heights_, k)
            {
                if (pp.coupled())
                {
                    addSide
                    (
                        k, own, nbrFraction[k][bFacei], nbrBin[bFacei],
                        nbrX[bFacei], 1, pSf[i], pCf[i], pPhi[i]
                    );
                }
                else if (fraction[k][own] > 0)
                {
                    add
                    (
                        k, bin[own], patchGroup[patchi], fraction[k][own],
                        pSf[i], pCf[i], pPhi[i], fluxf
                    );
                }
            }
        }
    }

    Pstream::listCombineReduce(sums, plusEqOp<scalar>());

    if (Pstream::master())
    {
        autoPtr<OFstream> filePtr(newFileAtTime("balance", time_.value()));
        OFstream& os = filePtr();
        os.precision(14);

        writeHeaderValue(os, "URef", URef_);
        writeHeaderValue(os, "pRef", pRef_);
        writeHeaderValue(os, "interpolate", Switch(interpolate_));
        writeCommented(os, "height bin xLow xHigh group area xMean yMean Ss Sn massIn");
        for (const surfaceVectorField& f : flux)
        {
            os << ' ' << f.name() << "_s " << f.name() << "_n";
        }
        os << nl;

        forAll(heights_, k)
        {
            for (label b = 0; b < nBins; ++b)
            {
                for (label g = 0; g < nGroups; ++g)
                {
                    const scalar* s = &sums[((k*nBins + b)*nGroups + g)*nSums];
                    if (s[AREA] <= 0)
                    {
                        continue;
                    }
                    os  << heights_[k] << ' ' << b << ' ' << edges_[b] << ' '
                        << edges_[b + 1] << ' ' << groupNames[g] << ' '
                        << s[AREA] << ' ' << s[AREA_S]/s[AREA] << ' '
                        << s[AREA_N]/s[AREA];
                    for (label i = S_S; i < nSums; ++i)
                    {
                        os << ' ' << s[i];
                    }
                    os << nl;
                }
            }
        }
    }

    Info<< type() << " " << name() << ": " << nHeights << " height(s), "
        << nBins << " bin(s) written" << nl << endl;

    return true;
}


// ************************************************************************* //
