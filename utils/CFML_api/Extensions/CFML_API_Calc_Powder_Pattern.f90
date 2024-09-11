module CFML_API_Calc_Powder_Pattern

use CFML_GlobalDeps,           only: to_Deg
use CFML_Math_General,         only: asind,locate
use CFML_Atom_TypeDef,         only: Atom_Type, Atom_List_Type,Allocate_Atom_List
use CFML_Reflections_Utilities,only: Reflection_List_Type
use CFML_Diffraction_Patterns, only: Diffraction_Pattern_Type, Allocate_Diffraction_Pattern
use CFML_PowderProfiles_CW,    only: TCH_pVoigt, PseudoVoigt, TCH
use CFML_IO_Formats,           only: Job_Info_type

implicit none

CONTAINS

Subroutine Calc_Powder_Pattern(Job_info,scalef,Hkl,Pat)
   !---- Argument ----!
   Type(Job_info_type),            intent(in)  :: Job_info
   Type(Reflection_List_Type),     intent(in)  :: hkl
   Type(Diffraction_Pattern_Type), intent(out) :: Pat
   real,                           intent(in)  :: Scalef

   !--- Local Variables ----!
   integer :: i,j,npts,i1,i2
   real    :: step,Intens,Bragg,Hl,Hg, ss,cs,tt,th1,th2,LorentzF, Y,eta,fwhm,chw
   real    :: thmin, thmax, thstep, lambda

   thmin = Job_info%range_2theta(1)%mina
   thmax = Job_info%range_2theta(1)%maxb
   thstep = Job_info%theta_step
   lambda = job_info%Lambda(1)%mina

   npts=(Thmax-Thmin)/thstep + 1.02

   call Allocate_Diffraction_Pattern(Pat,npts)
   
   Pat%Title=adjustl(Trim(Job_info%title))
   
   i=len_trim(Pat%Title)
   write(unit=Pat%Title(i+2:),fmt="(a,f7.4)") " => lambda: ", &
              Lambda
   !write(*,*) 'U,V,W,X,Y', Job_info%U, Job_info%V, Job_info%W, Job_info%X, Job_info%Y
   
   Pat%scat_var="2-Theta"
   Pat%instr="Calculated Pattern"
   Pat%xmin= Thmin
   Pat%xmax= Thmax
   Pat%ymin= 0.0
   Pat%ymax=0.0
   Pat%scal=1.0
   Pat%monitor=0.0
   Pat%step=Thstep
   Pat%Tsamp=300.0
   Pat%Tset=300.0
   Pat%npts=npts
   Pat%ct_step=.true.
   Pat%conv=(/Lambda,Lambda,0.0,0.0,0.0/)
   chw=15.0
   do i=1,npts
      Pat%x(i)=Pat%xmin+real(i-1)*Pat%step
   end do
      
   Pat%bgr(:)=Job_info%bkg

   do i=1,hkl%nref
      ss=Lambda*hkl%ref(i)%S !ss = sin(theta)
      cs=sqrt(abs(1.0-ss*ss))                  !cs = cos(theta)
      tt=ss/cs                                 !tt = tan(theta)
      LorentzF=0.5/(ss*ss*cs)
      Bragg=2.0*asind(ss)                      !Bragg = 2 * theta (in degrees)
            
      !fwhm_G = U*tan^2(theta) + V*tan(theta) + W
      HG=sqrt(tt*(Job_info%U*tt + Job_info%V) + Job_info%W)
      HL=Job_info%X*tt + Job_info%Y/cs

      !write(*,*) hg, hl
      call TCH(hg,hl,fwhm,eta)

      Select Case(nint(eta*10.0))
         Case(:2)
            chw=25.0
         case(3:5)
            chw=45.0
         case(6:7)
            chw=60.0
         case(8:)
            chw=90.0
      End Select

      th1=Bragg-chw*fwhm
      th2=Bragg+chw*fwhm
      
      i1=Locate(Pat%x,npts,th1)
      i2=Locate(Pat%x,npts,th2)
      i1=max(i1,1)
      i2=min(i2,npts)
      Intens= LorentzF *hkl%ref(i)%mult * hkl%ref(i)%Fc**2 * Scalef

      !write(*,*) Bragg, fwhm, eta
      
      do j=i1,i2
         !write(*,*) Pat%ycalc(j), PseudoVoigt( Pat%x(j)-Bragg, (/fwhm,eta /) ), Intens
         Pat%ycalc(j)=Pat%ycalc(j)+ PseudoVoigt( Pat%x(j)-Bragg, (/fwhm,eta /) ) * Intens
      end do
   end do


   
   return
End Subroutine Calc_Powder_Pattern

end module CFML_API_Calc_Powder_Pattern
