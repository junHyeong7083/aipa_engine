"""
시뮬레이션 서비스 (Simulation Service)

C#으로 비유하면 전체 파이프라인을 오케스트레이션하는 Mediator/Orchestrator 패턴.
개별 서비스들(생성기, 응답기, 캘리브레이터)을 조합해서 전체 흐름을 관리.

전체 흐름:
1. 페르소나 생성 (PopulationGenerator) → 가상 응답자 패널
2. 설문 실행 (ResponseGenerator) → 각 페르소나가 설문 응답
3. 캘리브레이션 (Calibrator) → 통계 보정
"""

from typing import Optional

from ..models.persona import Persona, PersonaConfig
from ..models.survey import SurveyQuestion, SurveyResponse
from .population_generator import PopulationGenerator  # 페르소나 생성기
from .response_generator import ResponseGenerator      # 설문 응답 생성기
from .calibrator import Calibrator                      # 통계 보정기
from .llm_service import LLMService                     # AI 텍스트 생성기


class SimulationService:
    """
    시뮬레이션 오케스트레이터 (C#의 public class SimulationService : ISimulationService)

    DI(Dependency Injection) 패턴: 생성자에서 각 서비스를 주입받음.
    C#의 services.AddScoped<ISimulationService, SimulationService>() 같은 것.
    """

    def __init__(
        self,
        population_generator: Optional[PopulationGenerator] = None,
        response_generator: Optional[ResponseGenerator] = None,
        calibrator: Optional[Calibrator] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """
        생성자 - 각 서비스 주입 (없으면 기본 인스턴스 생성)
        C#의 public SimulationService(IPopulationGenerator gen, IResponseGenerator resp, ...)
        """
        self.llm_service = llm_service or LLMService()
        self.population_generator = population_generator or PopulationGenerator()
        self.response_generator = response_generator or ResponseGenerator(self.llm_service)
        self.calibrator = calibrator or Calibrator()

    async def generate_personas(
        self,
        config: PersonaConfig,
        generate_backstories: bool = False,
    ) -> list[Persona]:
        """
        1단계: 페르소나 생성

        통계 분포 기반으로 가상 인물 패널을 만듦.
        generate_backstories=True이면 각 인물마다 AI가 배경 스토리도 작성.
        """
        # 인구 통계 기반 페르소나 생성
        personas = await self.population_generator.generate(config)

        # 배경 스토리 생성 (옵션)
        if generate_backstories:
            for persona in personas:
                persona.backstory = await self.llm_service.generate_backstory(persona)

        return personas

    async def run_survey(
        self,
        personas: list[Persona],
        questions: list[SurveyQuestion],
        generate_explanations: bool = True,
    ) -> list[SurveyResponse]:
        """
        2단계: 설문 실행

        모든 페르소나가 모든 질문에 답변.
        C#의 이중 foreach (foreach persona, foreach question) 패턴.

        예: 10명 x 5질문 = 50개 응답 생성
        """
        all_responses = []

        for persona in personas:
            for question in questions:
                # 각 페르소나-질문 조합에 대해 응답 생성
                response = await self.response_generator.generate_response(
                    persona,
                    question,
                    generate_explanation=generate_explanations,
                )
                all_responses.append(response)

        return all_responses

    async def calibrate(
        self,
        responses: list[SurveyResponse],
        config: PersonaConfig,
        personas: Optional[list[Persona]] = None,
    ) -> list[SurveyResponse]:
        """
        3단계: 캘리브레이션 (통계 보정)

        생성된 페르소나의 인구통계 분포가 목표와 다를 수 있으므로
        가중치를 조정해서 보정. (실제 여론조사에서 필수적으로 하는 과정)
        """
        if not personas:
            # 페르소나 정보 없으면 보정 불가 → 그대로 반환
            return responses

        return await self.calibrator.calibrate(
            personas,
            responses,
            config,
        )

    async def run_full_simulation(
        self,
        config: PersonaConfig,
        questions: list[SurveyQuestion],
        generate_backstories: bool = False,
        generate_explanations: bool = True,
        enable_calibration: bool = True,
    ) -> tuple[list[Persona], list[SurveyResponse]]:
        """
        전체 파이프라인 한방 실행 (1+2+3단계 통합)
        C#의 public async Task<(List<Persona>, List<SurveyResponse>)> RunFullAsync(...)

        간편하게 전체 흐름을 한번에 돌리고 싶을 때 사용.
        """
        # Step 1: 페르소나 생성
        personas = await self.generate_personas(
            config,
            generate_backstories=generate_backstories,
        )

        # Step 2: 설문 실행
        responses = await self.run_survey(
            personas,
            questions,
            generate_explanations=generate_explanations,
        )

        # Step 3: 캘리브레이션
        if enable_calibration:
            responses = await self.calibrate(responses, config, personas)

        # 튜플로 반환 (C#의 (personas, responses) 반환과 동일)
        return personas, responses
